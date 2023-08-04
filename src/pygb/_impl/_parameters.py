from dataclasses import dataclass


@dataclass
class GBParameters:
    sigma: float
    sigma_delta: float
    dim_act: int
    dim_obs: int
    len_sequence: int
    len_epoch: int
    epoch_sets: int

    def update(self, increment: "GBParameterIncrement") -> None:
        for attribute_name, _ in increment.__dataclass_fields__.items():
            if (increment_value := increment.__getattribute__(attribute_name)) is not None:
                self.__setattr__(attribute_name, increment_value)


@dataclass
class GBParameterIncrement:
    sigma: float | None = None
    sigma_delta: float | None = None
    dim_act: int | None = None
    dim_obs: int | None = None
    len_sequence: int | None = None
    len_epoch: int | None = None
    epoch_sets: int | None = None
