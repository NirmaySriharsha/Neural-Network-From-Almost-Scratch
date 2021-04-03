import unittest
from numpy.testing import assert_allclose
import numpy as np

import sys
sys.path.append("../")

import hw3_lib
import hw3_sol

TOLERANCE = 1e-8


class TestLinearLayer(unittest.TestCase):
  def setUp(self):
    np.random.seed(10701)
    batch_size = 10
    input_features = 20
    output_feature = 5
    self.x = np.random.uniform(-5, 5, (batch_size, input_features))
    self.grad = np.random.uniform(-2, 2, (batch_size, output_feature))
    self.params = hw3_sol.init_linear_params(input_features, output_feature)
    self.ref_data = np.load("tests/linear.npz")

  def test_forward(self):
    hw3_linear = hw3_lib.Linear(self.params)

    assert_allclose(hw3_linear.forward(self.x),
                    self.ref_data["fw"],
                    atol=TOLERANCE)

  def test_backward(self):
    hw3_linear = hw3_lib.Linear(self.params)

    hw3_linear.forward(self.x)

    assert_allclose(hw3_linear.backward(self.grad),
                    self.ref_data["bw"],
                    atol=TOLERANCE)


class TestReLU(unittest.TestCase):
  def setUp(self):
    np.random.seed(10701)
    batch_size = 10
    input_features = 20
    self.x = np.random.uniform(-5, 5, (batch_size, input_features))
    self.x[0, 0] = 0
    self.grad = np.random.uniform(-2, 2, (batch_size, input_features))
    self.ref_data = np.load("tests/relu.npz")

  def test_forward(self):
    hw3_relu = hw3_lib.ReLU()

    assert_allclose(hw3_relu.forward(self.x),
                    self.ref_data["fw"],
                    atol=TOLERANCE)

  def test_backward(self):
    hw3_relu = hw3_lib.ReLU()

    hw3_relu.forward(self.x)

    assert_allclose(hw3_relu.backward(self.grad),
                    self.ref_data["bw"],
                    atol=TOLERANCE)


class TestSigmoid(unittest.TestCase):
  def setUp(self):
    np.random.seed(10701)
    batch_size = 10
    input_features = 20
    self.x = np.random.uniform(-5, 5, (batch_size, input_features))
    self.grad = np.random.uniform(-2, 2, (batch_size, input_features))
    self.ref_data = np.load("tests/sigmoid.npz")

  def test_forward(self):
    hw3_sigmoid = hw3_lib.Sigmoid()

    assert_allclose(hw3_sigmoid.forward(self.x),
                    self.ref_data["fw"],
                    atol=TOLERANCE)

  def test_backward(self):
    hw3_sigmoid = hw3_lib.Sigmoid()

    hw3_sigmoid.forward(self.x)

    assert_allclose(hw3_sigmoid.backward(self.grad),
                    self.ref_data["bw"],
                    atol=TOLERANCE)


class TestGradientDescentOptimizer(unittest.TestCase):
  def setUp(self):
    np.random.seed(10701)
    batch_size = 10
    input_features = 20
    output_feature = 5
    self.learning_rate = 11
    self.x = np.random.uniform(-5, 5, (batch_size, input_features))
    self.grad = np.random.uniform(-2, 2, (batch_size, output_feature))
    self.params = hw3_sol.init_linear_params(input_features, output_feature)
    self.ref_data = np.load("tests/gd.npz")

  def test_step(self):
    hw3_model = hw3_sol.MultiLayerPerceptron([hw3_lib.Linear(self.params)])
    hw3_gd = hw3_sol.GradientDescentOptimizer(hw3_model, self.learning_rate)

    hw3_model.layers[0].forward(self.x)
    hw3_model.layers[0].backward(self.grad)
    hw3_gd.step()

    assert_allclose(hw3_model.layers[0].weights,
                    self.ref_data["w"],
                    atol=TOLERANCE,
                    err_msg="incorrect linear layer weights")

    assert_allclose(hw3_model.layers[0].biases,
                    self.ref_data["b"],
                    atol=TOLERANCE,
                    err_msg="incorrect linear layer biases")


class TestCrossEntropyLoss(unittest.TestCase):
  def setUp(self):
    np.random.seed(10701)
    batch_size = 10
    num_classes = 10
    self.y_hat = np.random.uniform(-5, 5, (batch_size, num_classes))
    self.y_true = np.array([7, 6, 4, 5, 1, 5, 2, 5, 8, 2])
    self.ref_data = np.load("tests/xeloss.npz")

  def test_forward(self):
    hw3_xeloss = hw3_lib.CrossEntropyLoss()
    assert_allclose(hw3_xeloss.forward(self.y_hat, self.y_true),
                    self.ref_data["fw"],
                    atol=TOLERANCE)

  def test_backward(self):
    hw3_xeloss = hw3_lib.CrossEntropyLoss()
    hw3_xeloss.forward(self.y_hat, self.y_true)

    assert_allclose(hw3_xeloss.backward(),
                    self.ref_data["bw"],
                    atol=TOLERANCE)


class TestTrainNetwork(unittest.TestCase):
  def setUp(self):
    np.random.seed(10701)
    self.data = hw3_sol.load_data()
    layer1_params = hw3_sol.init_linear_params(784, 256)
    layer2_params = hw3_sol.init_linear_params(256, 10)

    self.hw3_model = hw3_sol.MultiLayerPerceptron([hw3_lib.Linear(layer1_params),
                                                   hw3_lib.Sigmoid(),
                                                   hw3_lib.Linear(layer2_params)])

    self.hw3_result = hw3_sol.train_network(self.hw3_model, self.data, 40, 0.01, 50)
    self.ref_data = np.load("tests/train.npz", allow_pickle=True)

  def test_final_stats(self):
    for key, sol_value in self.ref_data["results"].item().items():
      assert_allclose(sol_value,
                      self.hw3_result[key],
                      atol=TOLERANCE,
                      err_msg=f"{key} mismatch")