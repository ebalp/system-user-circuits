"""Base class for system-user conflicts (system constraint vs contradicting user instruction)."""

from typing import Any, Callable


class Conflict:
    """One conflict: system constraint vs user instruction. Subclasses set conflict_id, templates, verifiers, arg_keys."""

    conflict_id: str = ""
    system_template: str = ""
    user_template: str = ""
    verify_system_fn: Callable[..., bool] = lambda r: False
    verify_user_fn: Callable[..., bool] = lambda r: False
    arg_keys: list[str] = []

    def __init__(self) -> None:
        self._instruction_args: dict[str, Any] | None = None
        self.conflict_id = getattr(self.__class__, "conflict_id", "")
        self._system_template = getattr(self.__class__, "system_template", "")
        self._user_template = getattr(self.__class__, "user_template", "")
        self._verify_system_fn = getattr(self.__class__, "verify_system_fn", lambda r: False)
        self._verify_user_fn = getattr(self.__class__, "verify_user_fn", lambda r: False)
        self._arg_keys = list(getattr(self.__class__, "arg_keys", None) or [])

    def _store_args(self, **kwargs: Any) -> None:
        """Store kwargs for get_instruction_args."""
        self._instruction_args = dict(kwargs)

    def build_system_prompt(self, **kwargs: Any) -> str:
        """Fill system_template with kwargs and return the string."""
        args = dict(kwargs)
        self._store_args(**args)
        out = self._system_template
        for k, v in args.items():
            out = out.replace("{" + str(k) + "}", str(v))
        return out

    def build_user_conflict_prompt(self, **kwargs: Any) -> str:
        """Fill user_template with stored or given kwargs and return the string."""
        args = self.get_instruction_args() or dict(kwargs)
        out = self._user_template
        for k, v in args.items():
            out = out.replace("{" + str(k) + "}", str(v))
        return out

    def get_instruction_args(self) -> dict[str, Any] | None:
        """Return args from last build."""
        return self._instruction_args

    def get_instruction_args_keys(self) -> list[str]:
        """Return keys accepted by build_* (for sampling)."""
        return list(self._arg_keys)

    def sample_args(self) -> dict[str, Any]:
        """Return random kwargs for templates. Override in subclasses with placeholders."""
        return {}

    def verify_followed_system(self, response: str) -> bool:
        """True if response satisfies the system constraint."""
        args = self.get_instruction_args() or {}
        if hasattr(self._verify_system_fn, "__code__") and self._verify_system_fn.__code__.co_argcount >= 2:
            return self._verify_system_fn(response, args)
        return self._verify_system_fn(response)

    def verify_followed_user(self, response: str) -> bool:
        """True if response satisfies the user (conflict) instruction."""
        args = self.get_instruction_args() or {}
        if hasattr(self._verify_user_fn, "__code__") and self._verify_user_fn.__code__.co_argcount >= 2:
            return self._verify_user_fn(response, args)
        return self._verify_user_fn(response)
