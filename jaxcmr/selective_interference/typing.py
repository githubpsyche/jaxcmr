"""Type definitions for selective interference simulations.

"""

from typing import Mapping, Optional, Protocol, runtime_checkable

from jaxtyping import Array, Bool, Float, Integer

Float_ = Float[Array, ""] | float | int
Int_ = Integer[Array, ""] | int


@runtime_checkable
class PhasedMemorySearch(Protocol):
    """Model of memory search with multi-phase encoding."""

    item_count: int
    is_active: Bool[Array, ""]
    recallable: Bool[Array, " item_count"]
    recall_total: Integer[Array, ""]
    study_index: Integer[Array, ""]

    def experience_film(self, choice: Int_) -> "PhasedMemorySearch":
        """Encode a film-phase item.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.

        Returns
        -------
        PhasedMemorySearch

        """
        ...

    def start_retrieving(self) -> "PhasedMemorySearch":
        """Return model after transitioning to retrieval mode.

        Returns
        -------
        PhasedMemorySearch

        """
        ...

    def retrieve(self, choice: Int_) -> "PhasedMemorySearch":
        """Return model after retrieving an item or stopping.

        Parameters
        ----------
        choice : Int_
            Index of item to retrieve (1-indexed). 0 terminates.

        Returns
        -------
        PhasedMemorySearch

        """
        ...

    def activations(self) -> Float[Array, " item_count"]:
        """Return retrieval support for each item.

        Returns
        -------
        Float[Array, " item_count"]

        """
        ...

    def outcome_probability(self, choice: Int_) -> Float[Array, ""]:
        """Return probability of a retrieval event.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed) or 0 to stop.

        Returns
        -------
        Float[Array, ""]

        """
        ...

    def outcome_probabilities(self) -> Float[Array, " recall_outcomes"]:
        """Return probabilities of all retrieval events.

        Returns
        -------
        Float[Array, " recall_outcomes"]

        """
        ...

    def experience_break(self, choice: Int_) -> "PhasedMemorySearch":
        """Encode a break-phase item.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.

        Returns
        -------
        PhasedMemorySearch

        """
        ...

    def experience_interference(self, choice: Int_) -> "PhasedMemorySearch":
        """Encode an interference-phase item.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.

        Returns
        -------
        PhasedMemorySearch

        """
        ...

    def experience_filler(self, choice: Int_) -> "PhasedMemorySearch":
        """Encode a filler-phase item.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.

        Returns
        -------
        PhasedMemorySearch

        """
        ...

    def start_reminders(self) -> "PhasedMemorySearch":
        """Transition to reminder phase.

        Returns
        -------
        PhasedMemorySearch

        """
        ...

    def remind(self, choice: Int_) -> "PhasedMemorySearch":
        """Replay a single item's context association without learning.

        Parameters
        ----------
        choice : Int_
            Index of item (1-indexed). 0 is ignored.

        Returns
        -------
        PhasedMemorySearch

        """
        ...


@runtime_checkable
class PhasedMemorySearchCreateFn(Protocol):
    """Factory for creating PhasedMemorySearch models."""

    def __call__(
        self,
        list_length: int,
        parameters: Mapping[str, Float_],
        connections: Optional[Float[Array, " study_events study_events"]],
    ) -> PhasedMemorySearch:
        """Create a new phased memory search model.

        Parameters
        ----------
        list_length : int
            Number of item slots in the model.
        parameters : Mapping[str, Float_]
            Model parameters.
        connections : Float[Array, " study_events study_events"] or None
            Optional pre-experimental association matrix.

        Returns
        -------
        PhasedMemorySearch

        """
        ...
