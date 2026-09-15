"""Cobaya's CLASS interface backed exclusively by hiclassy."""
from cobaya.log import LoggedError
from cobaya.theories.classy.classy import (
    classy,
    non_linear_default_code,
    non_linear_null_value,
)
from cobaya.theories.cosmo import BoltzmannBase


class HiClassTheory(classy):
    """Reuse Cobaya's observables and collectors with a HiClass instance."""

    def initialize(self):
        import hiclassy

        if self.path:
            raise LoggedError(
                self.log, "Install hiclassy in the Python environment; "
                "the legacy CLASS path option is not supported.")
        # These attribute names are part of the inherited Cobaya interface.
        self.classy_module = hiclassy
        self.classy = hiclassy.HiClass()
        BoltzmannBase.initialize(self)
        self.extra_args = dict(self.extra_args or {})
        self.extra_args.setdefault("output", "")
        if "non linear" in self.extra_args:
            if "non_linear" in self.extra_args:
                raise LoggedError(
                    self.log, "Specify only one of non_linear and non linear.")
            self.extra_args["non_linear"] = self.extra_args.pop("non linear")
        if self.extra_args.get("non_linear", "unset") in (None, False):
            self.extra_args["non_linear"] = non_linear_null_value
        elif ("non_linear" not in self.extra_args
              or self.extra_args["non_linear"] is True):
            self.extra_args["non_linear"] = non_linear_default_code
        self.derived_extra = []

    @classmethod
    def is_installed(cls, **kwargs):
        try:
            from hiclassy import HiClass
        except ImportError:
            return False
        return callable(HiClass)

    @classmethod
    def install(cls, **kwargs):
        """Never let the inherited installer download the legacy backend."""
        if cls.is_installed():
            return True
        raise RuntimeError("Install hiclassy in the Python environment first.")
