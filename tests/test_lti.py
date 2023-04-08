def test_dummy():
    assert True == True


from state_space_models.lti import SpringMassDamper


def test_spring_mass():
    s = SpringMassDamper.from_params()
    m = s.controllability
