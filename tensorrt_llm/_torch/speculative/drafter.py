from abc import ABC, abstractmethod
from typing import List, Optional, final

from ..pyexecutor.llm_request import LlmRequest, get_draft_token_length
from ..pyexecutor.resource_manager import ResourceManager
from ..pyexecutor.scheduler import ScheduledRequests


class Drafter(ABC):
    """Abstract base class for all drafter implementations."""

    def __init__(self,
                 max_concurrency: Optional[int] = None,
                 draft_len_schedule: Optional[dict[int, int]] = None) -> None:
        self.max_concurrency = max_concurrency
        # Store schedule as a sorted list of (threshold, value)
        if draft_len_schedule:
            self._draft_len_schedule = sorted(draft_len_schedule.items(),
                                              key=lambda kv: kv[0])
        else:
            self._draft_len_schedule = None

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
        Pad draft tokens to the max draft length for CUDA graph compatibility.

        Args:
            scheduled_requests: The scheduled requests to pad
        """
        for req in scheduled_requests.generation_requests:
            max_draft_tokens = self.max_draft_tokens
            num_draft_tokens = get_draft_token_length(req)
            req.py_draft_tokens.extend(
                0 for _ in range(max_draft_tokens - num_draft_tokens))

    # Helper to compute effective draft len from schedule, given current scheduled generation size
    def _effective_draft_len(self, scheduled_gen_size: int,
                             base_max_draft_len: int) -> int:
        if self._draft_len_schedule is None:
            return base_max_draft_len
        # Find first threshold >= scheduled_gen_size
        draft_len = self._draft_len_schedule[-1][1]
        for threshold, value in self._draft_len_schedule:
            if scheduled_gen_size <= threshold:
                draft_len = value
                break
        # Clamp to [0, base_max_draft_len]
        if draft_len < 0:
            draft_len = 0
        if draft_len > base_max_draft_len:
            draft_len = base_max_draft_len
        return draft_len
