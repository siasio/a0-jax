import chex
import jax
import jax.numpy as jnp
import pax
from typing import TypeVar, List

T = TypeVar("T", bound="Module")


class ResidualBlock(pax.Module):
    """A residual block of conv layers

    y(x) = x + F(x),

    where:
        F = BatchNorm >> relu >> Conv1 >> BatchNorm >> relu >> Conv2
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.batchnorm1 = pax.BatchNorm2D(dim, True, True)
        self.batchnorm2 = pax.BatchNorm2D(dim, True, True)
        self.conv1 = pax.Conv2D(dim, dim, 3)
        self.conv2 = pax.Conv2D(dim, dim, 3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        t = x
        t = jax.nn.relu(self.batchnorm1(t))
        t = self.conv1(t)
        t = jax.nn.relu(self.batchnorm2(t))
        t = self.conv2(t)
        return x + t


class ResnetPolicyValueNet(pax.Module):
    """Residual Conv Policy-Value network.

    Two-head network:
                        ┌─> action head
    input ─> Backbone  ─┤
                        └─> value head
    """

    def __init__(
        self, input_dims, num_actions: int, dim: int = 64, num_resblock: int = 5
    ) -> None:
        super().__init__()
        self.dim = dim
        if len(input_dims) == 3:
            num_input_channels = input_dims[-1]
            input_dims = input_dims[:-1]
            self.has_channel_dim = True
        else:
            num_input_channels = 1
            self.has_channel_dim = False

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.backbone = pax.Sequential(
            pax.Conv2D(num_input_channels, dim, 1), pax.BatchNorm2D(dim)
        )
        for _ in range(num_resblock):
            self.backbone >>= ResidualBlock(dim)
        self.action_head = pax.Sequential(
            pax.Conv2D(dim, dim, 1),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, self.num_actions, kernel_shape=input_dims, padding="VALID"),
        )
        self.value_head = pax.Sequential(
            pax.Conv2D(dim, dim, 1),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, dim, kernel_shape=input_dims, padding="VALID"),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, 1, kernel_shape=1, padding="VALID"),
            jnp.tanh,
        )

    def __call__(self, x: chex.Array, batched: bool = False):
        """Compute the action logits and value.

        Support both batched and unbatched states.
        """
        x = x.astype(jnp.float32)
        if not batched:
            x = x[None]  # add the batch dimension
        if not self.has_channel_dim:
            x = x[..., None]  # add channel dimension
        x = self.backbone(x)
        return x


class ResnetPolicyValueNet128(ResnetPolicyValueNet):
    """Create a resnet of 128 channels, 5 blocks"""

    def __init__(
        self, input_dims, num_actions: int, dim: int = 128, num_resblock: int = 5
    ):
        super().__init__(input_dims, num_actions, dim, num_resblock)


class ResnetPolicyValueNet256(ResnetPolicyValueNet):
    """Create a resnet of 256 channels, 6 blocks"""

    def __init__(
        self, input_dims, num_actions: int, dim: int = 256, num_resblock: int = 6
    ):
        super().__init__(input_dims, num_actions, dim, num_resblock)


class OwnershipHeadShared(pax.Module):
    def __init__(self, dim, input_dims, num_outputs):
        super().__init__()
        # Input is last layer of the ResNet concatenated with a mask
        self.backbone = pax.Sequential(
            pax.Conv2D(dim + 2, dim, 1),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
        )
        self.ownership_head = pax.Sequential(
            pax.Conv2D(dim, num_outputs, kernel_shape=input_dims, padding="VALID"),
            jnp.tanh,
        )
        self.policy_head = pax.Sequential(
            #pax.Conv2D(dim, dim, kernel_shape=input_dims, padding="VALID"),
            #pax.BatchNorm2D(dim, True, True),
            #jax.nn.relu,
            pax.Conv2D(dim, 2 * num_outputs + 1, kernel_shape=1, padding="VALID"),
        )

    def __call__(self, x, batched=False):
        x = self.backbone(x)
        action_logits = self.policy_head(x)
        ownership = self.ownership_head(x)

        if batched:
            return action_logits[:, 0, 0, :], ownership[:, 0, 0, :]
        else:
            return action_logits[0, 0, 0, :], ownership[0, 0, 0, :]


class OwnershipHead(pax.Module):
    def __init__(self, dim, input_dims, num_outputs, include_boardmask=False):
        super().__init__()
        # Input is last layer of the ResNet concatenated with a mask
        initial_dim = dim + 2 if include_boardmask else dim + 1
        self.ownership_head = pax.Sequential(
            pax.Conv2D(initial_dim, dim, 3),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, 1, 1, padding="VALID"),
            jnp.tanh,
        )
        #self.policy_head_old = pax.Sequential(
            # pax.Conv2D(dim, dim, kernel_shape=input_dims, padding="VALID"),
            # pax.BatchNorm2D(dim, True, True),
            # jax.nn.relu,
        #    pax.Conv2D(dim + 2, dim, 3),
        #    pax.BatchNorm2D(dim, True, True),
        #    jax.nn.relu,
        #    pax.Conv2D(dim, 2 * num_outputs + 1, kernel_shape=1, padding="VALID"),
        #)
        self.policy_head = pax.Sequential(
            #pax.Conv2D(dim, dim, kernel_shape=input_dims, padding="VALID"),
            #pax.BatchNorm2D(dim, True, True),
            #jax.nn.relu,
            pax.Conv2D(initial_dim, dim, 3),
            pax.BatchNorm2D(dim, True, True),
            jax.nn.relu,
            pax.Conv2D(dim, 2 * num_outputs + 1, kernel_shape=input_dims, padding="VALID"),
        )

    def __call__(self, x, batched=False):
        action_logits = self.policy_head(x)
        ownership = self.ownership_head(x)

        #if batched:
        #    return action_logits[:, 0, 0, :], ownership[:, 0, 0, :]
        #else:
        #    return action_logits[0, 0, 0, :], ownership[0, 0, 0, :]
        if batched:
            return jnp.squeeze(action_logits), jnp.squeeze(ownership)
        else:
            return jnp.squeeze(action_logits[0, ...]), jnp.squeeze(ownership[0, ...])


class TransferResnet(pax.Module):
    # backbone: ResnetPolicyValueNet
    # head: OwnershipHead

    def __init__(self, backbone: ResnetPolicyValueNet, head: OwnershipHead = None, input_dims=(19, 19), include_boardmask=False):
        super().__init__()
        # input_dims = backbone.input_dims
        if len(input_dims) == 3:
            input_dims = input_dims[:-1]
            self.has_channel_dim = True
        else:
            self.has_channel_dim = False
        self.module_dict = {}
        self.module_dict["backbone"] = backbone
        dim = self.module_dict["backbone"].dim
        self.num_intersections = input_dims[0] * input_dims[1]
        self.module_dict["head"] = OwnershipHead(dim=dim, input_dims=input_dims, num_outputs=self.num_intersections, include_boardmask=include_boardmask) if head is None else head
        self.include_boardmask = include_boardmask

    #parameters = pax.parameters_method("head", "backbone")

    def __call__(self, input: List[chex.Array], batched: bool = False):
        x, mask, board_mask = input
        x = x.astype(jnp.float32)
        mask = mask.astype(jnp.float32)
        board_mask = board_mask.astype(jnp.float32)
        x = self.module_dict["backbone"](x, batched=batched)
        mask = mask[..., None]
        if not batched:
            mask = mask[None]
        board_mask = board_mask[..., None]
        if not batched:
            board_mask = board_mask[None]
        x = jnp.concatenate((x, mask), axis=-1)
        if self.include_boardmask:
            x = jnp.concatenate((x, board_mask), axis=-1)
        return self.module_dict["head"](x, batched=batched)

    #def train(self: T) -> T:
    #    self.apply(lambda mod: mod.replace(_training=True)) #super().train()
    #    self.backbone = self.backbone.eval() #self.head.apply(lambda mod: mod.replace(_training=True))
    #    return self
