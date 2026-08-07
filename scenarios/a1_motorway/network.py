import casadi as cs
from sym_metanet import (
    Destination,
    Link,
    MainstreamOrigin,
    MeteredOnRamp,
    Network,
    Node,
)
from macroscopic.vsl import CarlsonLinkWithVsl
from macroscopic.simulation.parameters import BaseParameters


def build_network(parameters: BaseParameters, control_type: str) -> Network:
    '''
    Topology:
        O1 -- L1 --> N2 -- L2 --> N3
        O2 -- L2 --> N2
    '''
    N1 = Node(name="N1")
    N2 = Node(name="N2")
    N3 = Node(name="N3")

    O1 = MainstreamOrigin[cs.SX](name="O1")
    O2 = MeteredOnRamp[cs.SX](parameters.C[1], name="O2")
    D1 = Destination[cs.SX](name="D1")

    _common = dict(
        nb_segments=4,
        lanes=parameters.lanes,
        length=parameters.L,
        maximum_density=parameters.rho_max,
        critical_density=parameters.rho_crit,
        free_flow_velocity=parameters.v_free,
        name="L1",
    )

    if control_type in ("vsl", "coordinated"):
        from macroscopic.simulation.parameters import VslParameters
        assert isinstance(parameters, VslParameters)
        L1 = CarlsonLinkWithVsl(
            **_common,
            a=parameters.a,
            segments_with_vsl=parameters.segments_with_vsl,
            alpha=parameters.alpha_compliance,
            V=parameters.V,
            A=parameters.A,
            E=parameters.E,
        )
    else:
        L1 = Link[cs.SX](
            **_common,
            a=parameters.a,
        )

    L2 = Link[cs.SX](
        nb_segments=2,
        lanes=parameters.lanes,
        length=parameters.L,
        maximum_density=parameters.rho_max,
        critical_density=parameters.rho_crit,
        free_flow_velocity=parameters.v_free,
        a=parameters.a,
        name="L2",
    )

    return (
        Network(name="A1")
        .add_path(origin=O1, path=(N1, L1, N2, L2, N3), destination=D1)
        .add_origin(O2, N2)
    )
