"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .ntm import NTM
from .controller import LSTMController, FFWController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory


class EncapsulatedNTM(nn.Module):

    def __init__(
        self,
        num_inputs,
        num_outputs,
        controller_size,
        controller_layers,
        num_heads,
        N,
        M,
        controller_type
    ):
        """Initialize an EncapsulatedNTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        """
        super(EncapsulatedNTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M

        # Create the NTM components
        memory = NTMMemory(N, M)
        if controller_type == 'lstm-ntm':
            controller = LSTMController(
                num_inputs + M * num_heads,
                controller_size,
                controller_layers
            )
        elif controller_type == 'ffw-ntm':
            controller = FFWController(
                num_inputs + M * num_heads,
                controller_size
            )
        else:
            raise Exception(
                'No such controller. Choose either lstm-ntm or ffw-ntm'
            )

        heads = nn.ModuleList([])
        for i in range(num_heads):
            heads += [
                NTMReadHead(memory, controller_size),
                NTMWriteHead(memory, controller_size)
            ]

        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads)
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)

    def forward(self, x=None):
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.num_inputs))

        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params


class BaselineLSTM(nn.Module):
    """
        Basic LSTM Network. Adapted the EncapsulatedNTM to use only the
        LSTMController and no memory
    """
    def __init__(
        self,
        num_inputs,
        num_outputs,
        controller_size,
        controller_layers
    ):
        """Initialize a BaselineLSTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        """
        super(BaselineLSTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_layers = controller_layers

        self.controller = LSTMController(
            num_inputs,
            controller_size,
            controller_layers
        )

        # Take the output of the LSTM (hidden state) and pass it through
        # a Linear activation to output the copied input
        self.hidden2out = nn.Linear(controller_size, num_outputs)

    def enable_cuda(self):
        self.cuda()
        self.controller.cuda()
        print('Cuda Enabled!')

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.previous_state = self.controller.create_new_state(batch_size)

    def forward(self, x=None):
        # If x is None it's because we don't want anything in the input
        # So putting the input to zeros.
        if x is None:
            x = Variable(torch.zeros(self.batch_size, self.num_inputs))

        o, self.previous_state = self.controller(x, self.previous_state)
        return F.sigmoid(self.hidden2out(o)), self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
