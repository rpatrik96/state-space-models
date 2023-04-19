from state_space_models.lti import SpringMassDamper, LTISystem


def test_spring_mass():
    s = SpringMassDamper.from_params()
    m = s.controllability
