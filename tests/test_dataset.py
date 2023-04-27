from state_space_models.dataset import LTIDataset


def test_lti_dataset(lti_AB):
    """Creates an LTISystem then initializes the LTIDataset with it."""
    dataset = LTIDataset(lti_AB, 50, 0.1)
