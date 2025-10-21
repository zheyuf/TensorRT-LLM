from abc import ABC, abstractmethod
from bisect import bisect_right
from typing import Dict, List, Optional, final

from tensorrt_llm.logger import logger

from ..pyexecutor.llm_request import LlmRequest, get_draft_token_length
from ..pyexecutor.resource_manager import ResourceManager
from ..pyexecutor.scheduler import ScheduledRequests


class Drafter(ABC):
    """Abstract base class for all drafter implementations."""

    def __init__(
        self,
        max_draft_tokens: int,
        max_concurrency: Optional[int] = None,
        draft_len_schedule: Optional[Dict[int, int]] = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        # Schedule is already validated and sorted by config validator
        self.draft_len_schedule = draft_len_schedule
        # It's dynamic if draft_len_schedule is provided in spec_config (dynamic draft length based on runtime batch size is enabled). It's static in other cases.
        self.max_draft_tokens = max_draft_tokens
        # It's always static
        self._static_max_draft_tokens = max_draft_tokens

    @abstractmethod
    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        """
        Prepare the drafter tokens for the forward computation this step.

        Args:
            scheduled_requests: The scheduled requests for this iteration
        """
        raise NotImplementedError

    @final
    def get_draft_len_for_batch_size(self, batch_size: int) -> int:
        """
        Get the appropriate draft length for the given batch size using binary search.

        Args:
            batch_size: Current batch size (has been sorted by config validator)

        Returns:
            The draft length to use for this batch size
        """

        # Binary search to find the largest threshold <= batch_size
        # draft_len_schedule is already sorted by config validator
        thresholds = list(self.draft_len_schedule.keys())

        # bisect_right finds where to insert batch_size to keep list sorted
        # The element before insertion point is the largest threshold <= batch_size
        idx = bisect_right(thresholds, batch_size)

        if idx == 0:
            # batch_size is smaller than smallest threshold (batch_size smaller than 1)
            # This shouldn't happen in practice, but handle defensively
            logger.warning(
                f"get_draft_len_for_batch_size called with batch_size={batch_size} < 1. "
                f"This is unexpected. Disabling speculation (returning draft_len=0)."
            )
            return 0

        # Return draft_len for the largest threshold <= batch_size
        threshold = thresholds[idx - 1]
        return self.draft_len_schedule[threshold]

    @final
    def should_use_spec_decode(self, requests: List[LlmRequest],
                               max_batch_size: int, max_num_tokens: int,
                               max_draft_len: int) -> bool:
        """
        You probably don't want to override this. ModelEngine
        assumes that speculation is always on if max_concurrency
        is not specified by the user's spec config.
        """

        # Inputs typically validated upstream: max_batch_size>0, max_num_tokens>0, max_draft_len>=0

        if self.max_concurrency is None:
            return True

        # Defensive guards; keep behavior explicit for zero/empty cases
        if not requests or max_batch_size <= 0 or max_num_tokens <= 0:
            return False

        tokens_per_request = 1 + max_draft_len
        token_cap = max_num_tokens // tokens_per_request
        if token_cap <= 0:
            return False

        num_effective_requests = min(len(requests), max_batch_size, token_cap)
        return num_effective_requests <= self.max_concurrency

    @final
    def pad_draft_tokens_for_cuda_graph(
            self, scheduled_requests: ScheduledRequests) -> None:
        """
        Pad draft tokens for CUDA graph compatibility.

        CUDA graphs require all requests in a batch to have the same tensor shape.
        Individual requests may generate fewer draft tokens (e.g., NGram mismatches,
        early stopping), but all must be padded to the same length.

        Stage 1 (No schedule): Pads to STATIC max_draft_tokens
        - Single CUDA graph captured with max_draft_len
        - All requests padded to _static_max_draft_tokens regardless of actual generation

        Stage 2 (With schedule): Pads to DYNAMIC max_draft_tokens (self.max_draft_tokens)
        - Multiple CUDA graphs captured for each unique draft length
        - All requests padded to current max_draft_tokens (which varies by batch size)
        - Example: With batch_size=4, max_draft_tokens=2, all requests padded to 2
          (even if some requests only generated 1 or 0 tokens)

        Args:
            scheduled_requests: The scheduled requests to pad
        """
        for req in scheduled_requests.generation_requests:
            num_draft_tokens = get_draft_token_length(req)

            if self.draft_len_schedule is not None:
                # Stage 2: Pad to current (dynamic) max_draft_tokens
                # This is the draft length for the current batch size
                target_len = self.max_draft_tokens
            else:
                # Stage 1: Pad to static max_draft_tokens
                target_len = self._static_max_draft_tokens

            # Pad if needed
            if num_draft_tokens < target_len:
                req.py_draft_tokens.extend(
                    0 for _ in range(target_len - num_draft_tokens))

    def run_drafter_post(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
        is_warmup: bool = False,
    ) -> None:
        """
        If draft forward needs to be run directly after the target model forward,
        this method can be overridden to do that.
        Used in SaveHiddenStatesDrafter (to ensure correct input_ids)
        """

    def update_max_draft_tokens(self, new_max_draft_tokens: int) -> None:
        """
        Update the dynamic max_draft_tokens based on current batch size.

        Used when draft_len_schedule is provided in spec_config (dynamic draft length
        based on runtime batch size is enabled). This updates the drafter's max_draft_tokens
        which determines how many draft tokens to generate and pad to.

        Note: This is the DYNAMIC value that changes per iteration. The STATIC maximum
        is stored in _static_max_draft_tokens and never changes.

        Subclasses can override to propagate to their resource managers if needed.

        Args:
            new_max_draft_tokens: The new max draft tokens for the current batch size
        """
        self.max_draft_tokens = new_max_draft_tokens
