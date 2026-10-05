"""Real CPU autocast checks for the portable Skynet execution precision helper."""
import unittest

from skynet_app.adapters.unidex_precision import (
    apply_policy_precision, precision_recipe, training_precision_kwargs, validate_precision_recipe,
)

try:
    import torch
except ImportError:
    torch = None


class PrecisionRecipeTests(unittest.TestCase):
    def test_auto_preserves_prior_recipe_shape(self):
        expected = {
            "schema": "skynet.unidex-execution-precision/v1", "precision": "fp32",
            "float32_matmul_precision": "highest", "geometry": "native", "input_dtype": "float32",
            "embedding_output_dtype": "native", "fsdp_mixed_precision": None, "evaluation_precision": "fp32",
        }
        self.assertEqual(precision_recipe(), expected)
        self.assertEqual(precision_recipe(fsdp_parameter_dtype="auto"), expected)
        for precision, matmul in [("fp32", "highest"), ("fp32", "high"), ("bf16", "highest")]:
            recipe = precision_recipe(precision, matmul, "auto")
            self.assertEqual(recipe, precision_recipe(precision, matmul))
            self.assertNotIn("fsdp_parameter_dtype", recipe)
            self.assertEqual(validate_precision_recipe(recipe), recipe)

    def test_fp32_parameters_require_explicit_bf16_recipe(self):
        original = precision_recipe("bf16")
        recipe = precision_recipe("bf16", fsdp_parameter_dtype="float32")
        self.assertEqual(recipe["fsdp_parameter_dtype"], "float32")
        self.assertEqual(recipe["fsdp_mixed_precision"]["param_dtype"], "float32")
        self.assertEqual(validate_precision_recipe(recipe), recipe)
        self.assertEqual(original["fsdp_mixed_precision"]["param_dtype"], "bfloat16")
        unpinned = {key: value for key, value in recipe.items() if key != "fsdp_parameter_dtype"}
        with self.assertRaises(ValueError):
            validate_precision_recipe(unpinned)
        for precision, dtype in [("fp32", "float32"), ("bf16", "bfloat16"), ("bf16", None)]:
            with self.subTest(precision=precision, dtype=dtype), self.assertRaises(ValueError):
                precision_recipe(precision, fsdp_parameter_dtype=dtype)

    def test_recipe_is_explicit_and_independent(self):
        recipe = precision_recipe("bf16")
        self.assertEqual(recipe["evaluation_precision"], "fp32")
        self.assertFalse(recipe["fsdp_mixed_precision"]["cast_root_forward_inputs"])
        changed = validate_precision_recipe(recipe)
        changed["fsdp_mixed_precision"]["buffer_dtype"] = "bfloat16"
        self.assertEqual(recipe["fsdp_mixed_precision"]["buffer_dtype"], "float32")
        with self.assertRaises(ValueError):
            validate_precision_recipe(changed)

    def test_rejects_undeclared_execution_modes(self):
        for args in [("fp16", "highest"), ("bf16", "high"), ("fp32", "medium")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                precision_recipe(*args)
        with self.assertRaises(ValueError):
            validate_precision_recipe({**precision_recipe(), "extra": True})


if torch is not None:
    class NativeGrouping(torch.nn.Module):
        """Native squared-distance/topk operations, with fixed FPS centers.

        Real CUDA FPS parity belongs to the GPU benchmark. CPU autocast tests
        the susceptible native KNN calculation without needing PyTorch3D.
        """
        def forward(self, xyz, colors):
            centers = xyz[:, :2]
            distances = -2 * torch.matmul(centers, xyz.transpose(1, 2))
            distances += torch.sum(centers ** 2, -1).unsqueeze(-1)
            distances += torch.sum(xyz ** 2, -1).unsqueeze(1)
            neighbors = torch.topk(distances, 3, dim=-1, largest=False, sorted=False).indices
            return distances, neighbors, centers


    class TinyNativePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pointcloud_encoder = torch.nn.Module()
            self.pointcloud_encoder.group_divider = NativeGrouping()
            self.embed_tokens = torch.nn.Embedding(4, 3)
            self.multi_modal_projector = torch.nn.Linear(3, 3)

        def assemble(self, pointcloud):
            # Same native indexed assembly and its input-driven destination dtype.
            final = torch.zeros(1, 2, 3, dtype=pointcloud.dtype)
            text = self.embed_tokens(torch.tensor([[1]]))
            cloud = self.multi_modal_projector(pointcloud[:, :1])
            final[:, [0]] = text
            final[:, [1]] = cloud
            return final


@unittest.skipIf(torch is None, "Torch is required for real autocast verification")
class PrecisionTorchTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.get_float32_matmul_precision()
        torch.manual_seed(9)

    def tearDown(self):
        torch.set_float32_matmul_precision(self.previous)

    def test_default_does_not_change_forward_hooks_or_state(self):
        policy = TinyNativePolicy()
        group_forward = policy.pointcloud_encoder.group_divider.forward
        state = {key: value.clone() for key, value in policy.state_dict().items()}
        apply_policy_precision(policy, precision_recipe())
        self.assertEqual(policy.pointcloud_encoder.group_divider.forward, group_forward)
        self.assertEqual(training_precision_kwargs(precision_recipe()), {})
        self.assertFalse(policy.embed_tokens._forward_hooks)
        self.assertFalse(policy.multi_modal_projector._forward_hooks)
        self.assertEqual(state.keys(), policy.state_dict().keys())
        for key, value in state.items():
            self.assertTrue(torch.equal(value, policy.state_dict()[key]))

    def test_geometry_matches_fp32_inside_real_bf16_autocast(self):
        policy = TinyNativePolicy()
        xyz = torch.randn(2, 31, 3) * .003 + .5
        colors = torch.rand_like(xyz)
        group = policy.pointcloud_encoder.group_divider
        reference = group(xyz, colors)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            raw = group(xyz, colors)
        self.assertFalse(torch.equal(reference[0], raw[0]))
        self.assertFalse(torch.equal(reference[1], raw[1]))
        apply_policy_precision(policy, precision_recipe("bf16"))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = group(xyz, colors)
        for expected, observed in zip(reference, actual):
            self.assertTrue(torch.equal(expected, observed))
        with self.assertRaisesRegex(ValueError, "lost their original FP32"):
            group(xyz.bfloat16(), colors.bfloat16())

    def test_geometry_restores_matmul_mode_even_after_failure(self):
        policy = TinyNativePolicy()
        observed = []
        def failing_group(xyz, colors):
            observed.append(torch.get_float32_matmul_precision())
            raise RuntimeError("group failed")
        policy.pointcloud_encoder.group_divider.forward = failing_group
        apply_policy_precision(policy, precision_recipe("fp32", "high"))
        with self.assertRaisesRegex(RuntimeError, "group failed"):
            policy.pointcloud_encoder.group_divider(torch.zeros(1, 3, 3), torch.zeros(1, 3, 3))
        self.assertEqual(observed, ["highest"])
        self.assertEqual(torch.get_float32_matmul_precision(), "high")

    def test_bf16_embedding_alignment_keeps_keys_and_gradients(self):
        policy = TinyNativePolicy()
        policy.embed_tokens.bfloat16()  # Emulate its separately wrapped FSDP unit.
        pointcloud = torch.randn(1, 2, 3)
        with self.assertRaisesRegex(RuntimeError, "dtypes match"):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                policy.assemble(pointcloud)
        keys = list(policy.state_dict())
        group_class = type(policy.pointcloud_encoder.group_divider)
        class_forward = group_class.forward
        recipe = precision_recipe("bf16")
        apply_policy_precision(policy, recipe)
        apply_policy_precision(policy, recipe)  # Idempotent installation.
        self.assertEqual(len(policy.embed_tokens._forward_hooks), 1)
        self.assertEqual(list(policy.state_dict()), keys)
        self.assertIs(group_class.forward, class_forward)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assembled = policy.assemble(pointcloud)
            loss = (assembled - torch.zeros_like(assembled)).square().mean()
        self.assertEqual(assembled.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        for parameter in policy.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        with self.assertRaisesRegex(ValueError, "cannot change"):
            apply_policy_precision(policy, precision_recipe())

    def test_bf16_fsdp_preserves_inputs_buffers_and_gradient_reduction(self):
        for option, expected in [("auto", torch.bfloat16), ("float32", torch.float32)]:
            with self.subTest(fsdp_parameter_dtype=option):
                recipe = precision_recipe("bf16", fsdp_parameter_dtype=option)
                mixed = training_precision_kwargs(recipe)["mixed_precision"]
                self.assertEqual(mixed.param_dtype, expected)
                self.assertEqual(mixed.reduce_dtype, torch.float32)
                self.assertEqual(mixed.buffer_dtype, torch.float32)
                self.assertFalse(mixed.cast_forward_inputs)
                self.assertFalse(mixed.cast_root_forward_inputs)
                self.assertFalse(mixed.keep_low_precision_grads)

    def test_bf16_compute_with_fp32_parameters_keeps_native_assembly_valid(self):
        policy = TinyNativePolicy()
        recipe = precision_recipe("bf16", fsdp_parameter_dtype="float32")
        apply_policy_precision(policy, recipe)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = policy.assemble(torch.randn(1, 2, 3))
            loss = result.square().mean()
        loss.backward()
        self.assertEqual(result.dtype, torch.float32)
        for parameter in policy.parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertEqual(parameter.grad.dtype, torch.float32)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_bf16_batchnorm_keeps_fp32_parameters_and_running_statistics(self):
        mixed = training_precision_kwargs(precision_recipe("bf16"))["mixed_precision"]
        batchnorm = torch.nn.BatchNorm1d(8)
        # FSDP's existing auto-wrap policy isolates these modules with mixed
        # precision disabled. BF16 weights plus FP32 running stats are invalid.
        self.assertIsInstance(batchnorm, tuple(mixed._module_classes_to_ignore))
        values = torch.randn(4, 8, 64).bfloat16().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = batchnorm(values)
        output.float().square().mean().backward()
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(batchnorm.weight.dtype, torch.float32)
        self.assertEqual(batchnorm.running_mean.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(values.grad).all())


def test_acceleration_identity_is_pinned_and_default_is_unchanged(tmp_path):
    from types import SimpleNamespace
    import pytest
    from skynet_app.adapters.unidex_runtime import (
        execution_precision, execution_sharding, training_identity, stable_digest, validate_resume,
    )
    from test_unidex_runtime import recording_fixture
    _, metadata = recording_fixture(tmp_path, count=2)
    selections = [dict(position=0, version_id='v', manifest_sha256='a' * 64, metadata=metadata)]
    args = SimpleNamespace(precision='fp32', batch_size=4, learning_rate=1e-4, num_workers=1,
                           seed=1701, gpu_count=4, gradient_accumulation=8)
    native = training_identity(args, {}, {}, {}, selections=selections)
    args.float32_matmul_precision = 'highest'
    assert native == training_identity(args, {}, {}, {}, selections=selections)
    assert execution_precision(args) is None and 'execution_precision' not in native
    args.float32_matmul_precision = 'high'
    accelerated = training_identity(args, {}, {}, {}, selections=selections)
    assert accelerated['execution_precision'] == precision_recipe('fp32', 'high')
    saved = {'skynet': {**accelerated, 'identity_sha256': stable_digest(accelerated)},
             'optimizer_states': [{}], 'lr_schedulers': [{}],
             'loops': {'fit_loop': {'state_dict': {'combined_loader': [{'schema': 'skynet.unidex-loader-state/v1'}]}}}}
    validate_resume(saved, accelerated)
    with pytest.raises(ValueError, match='identical'):
        validate_resume(saved, native)
    args.precision, args.float32_matmul_precision = 'bf16', 'highest'
    mixed = training_identity(args, {}, {}, {}, selections=selections)
    assert mixed['execution_precision']['fsdp_mixed_precision']['reduce_dtype'] == 'float32'
    with pytest.raises(ValueError, match='identical'):
        validate_resume(saved, mixed)

    args.distributed_strategy = 'fsdp'
    original_fsdp = training_identity(args, {}, {}, {}, selections=selections)
    args.fsdp_parameter_dtype, args.fsdp_sharding_strategy = 'auto', 'FULL_SHARD'
    assert training_identity(args, {}, {}, {}, selections=selections) == original_fsdp
    assert execution_sharding(args) is None and 'execution_sharding' not in original_fsdp
    args.fsdp_sharding_strategy = 'SHARD_GRAD_OP'
    deferred = training_identity(args, {}, {}, {}, selections=selections)
    assert deferred['execution_sharding'] == {
        'sharding_strategy': 'SHARD_GRAD_OP', 'gradient_sync': 'optimizer_boundary',
        'accumulation': 'fsdp_no_sync',
    }
    saved['skynet'] = {**deferred, 'identity_sha256': stable_digest(deferred)}
    validate_resume(saved, deferred)
    with pytest.raises(ValueError, match='identical'):
        validate_resume(saved, original_fsdp)

    args.fsdp_parameter_dtype = 'float32'
    fp32_parameters = training_identity(args, {}, {}, {}, selections=selections)
    assert fp32_parameters['execution_precision'] == precision_recipe('bf16', fsdp_parameter_dtype='float32')
    assert fp32_parameters['datasets'] == deferred['datasets']
    assert fp32_parameters['model_config'] == deferred['model_config']
    assert fp32_parameters['training'] == deferred['training']
    with pytest.raises(ValueError, match='identical'):
        validate_resume(saved, fp32_parameters)


def test_acceleration_reaches_runtime_through_shared_adapter_flags():
    from skynet_app.adapters import _apply_parameter_flags
    from skynet_app.adapters.unidex_manifest import manifest
    from skynet_app.adapters.unidex_runtime import parser, execution_precision, execution_sharding
    declared = manifest()
    path = 'native.config.float32_matmul_precision'
    flags, blockers = [], []
    _apply_parameter_flags(flags, {'native': {'config': {'float32_matmul_precision': 'high'}}},
                           {path: declared.train.parameter_flags[path]}, blockers)
    assert not blockers
    args = parser().parse_args(['--repository', '/repo', '--dataset', '/data', '--output', '/out', *flags])
    assert execution_precision(args)['geometry'] == 'fp32_guard'
    assert 'adapter-support/unidex_precision.py' in declared.train.capsule_files

    paths = ['train.precision', *['native.config.' + key for key in (
        'distributed_strategy', 'fsdp_sharding_strategy', 'fsdp_parameter_dtype',
    )]]
    for parameter_dtype in ('auto', 'float32'):
        flags, blockers = [], []
        _apply_parameter_flags(flags, {
            'train': {'precision': 'bf16'},
            'native': {'config': {'distributed_strategy': 'fsdp', 'fsdp_sharding_strategy': 'SHARD_GRAD_OP',
                                  'fsdp_parameter_dtype': parameter_dtype}},
        }, {path: declared.train.parameter_flags[path] for path in paths}, blockers)
        assert not blockers
        args = parser().parse_args(['--repository', '/repo', '--dataset', '/data', '--output', '/out', *flags])
        assert args.distributed_strategy == 'fsdp'
        assert execution_precision(args) == precision_recipe('bf16', fsdp_parameter_dtype=parameter_dtype)
        assert execution_sharding(args)['accumulation'] == 'fsdp_no_sync'


def test_fsdp_overrides_reject_non_fsdp_before_model_initialization():
    import pytest
    from skynet_app.adapters.unidex_runtime import parser, execution_precision, execution_sharding, training_strategy
    base = ['--repository', '/repo', '--dataset', '/data', '--output', '/out']
    args = parser().parse_args(base)
    assert execution_precision(args) is None
    assert execution_sharding(args) is None
    assert training_strategy('ddp', 1) == 'auto'
    assert training_strategy('ddp', 4) == 'ddp_find_unused_parameters_true'

    args = parser().parse_args([*base, '--fsdp-sharding-strategy', 'SHARD_GRAD_OP'])
    with pytest.raises(ValueError, match='FSDP sharding requires FSDP training'):
        execution_sharding(args)
    with pytest.raises(ValueError, match='FSDP sharding requires FSDP training'):
        training_strategy('ddp', 4, sharding_strategy=args.fsdp_sharding_strategy)
    args = parser().parse_args([*base, '--precision', 'bf16', '--fsdp-parameter-dtype', 'float32'])
    with pytest.raises(ValueError, match='FSDP parameter dtype requires FSDP training'):
        execution_precision(args)


def test_opt_in_fsdp_strategy_delegates_complete_accumulation_scope(monkeypatch):
    """Check the Lightning boundary with stand-ins, without a GPU/process group.

    Actual collective correctness and memory are checked by the distributed
    benchmark; this checks that only the opt-in strategy enters root no_sync
    and that a failed accumulation closure also leaves that context.
    """
    import contextlib
    import sys
    from types import ModuleType, SimpleNamespace
    import pytest
    from skynet_app.adapters import unidex_runtime as runtime

    class Strategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = None

    class Root:
        active = False
        entries = 0

        @contextlib.contextmanager
        def no_sync(self):
            self.active = True
            self.entries += 1
            try:
                yield
            finally:
                self.active = False

    strategies = ModuleType('pytorch_lightning.strategies')
    strategies.FSDPStrategy = Strategy
    fsdp = ModuleType('torch.distributed.fsdp')
    fsdp.FullyShardedDataParallel = Root
    monkeypatch.setitem(sys.modules, strategies.__name__, strategies)
    monkeypatch.setitem(sys.modules, fsdp.__name__, fsdp)
    mixed = object()
    seen = []

    def precision_kwargs(recipe):
        seen.append(recipe)
        return {'mixed_precision': mixed}

    monkeypatch.setattr(runtime, 'training_precision_kwargs', precision_kwargs)
    policy = SimpleNamespace(embed_tokens=object(), joint_model=SimpleNamespace(mixtures={}),
                             pointcloud_encoder=SimpleNamespace(visual=SimpleNamespace(blocks=[])))
    original = runtime.training_strategy('fsdp', 4, policy)
    assert type(original) is Strategy
    assert 'mixed_precision' not in original.kwargs

    recipe = precision_recipe('bf16')
    strategy = runtime.training_strategy('fsdp', 4, policy, execution_precision=recipe,
                                         sharding_strategy='SHARD_GRAD_OP')
    assert seen == [recipe]
    assert strategy.kwargs['mixed_precision'] is mixed
    assert strategy.kwargs['sharding_strategy'] == 'SHARD_GRAD_OP'
    assert strategy.kwargs['state_dict_type'] == original.kwargs['state_dict_type'] == 'full'
    assert strategy.kwargs['use_orig_params'] == original.kwargs['use_orig_params'] is True
    assert strategy.kwargs['auto_wrap_policy'] == original.kwargs['auto_wrap_policy']
    root = Root()
    strategy.model = root
    with strategy.block_backward_sync():
        assert root.active  # The caller performs both forward and backward here.
    assert not root.active and root.entries == 1
    with pytest.raises(RuntimeError, match='backward failed'):
        with strategy.block_backward_sync():
            assert root.active
            raise RuntimeError('backward failed')
    assert not root.active and root.entries == 2
    strategy.model = object()
    with pytest.raises(TypeError, match='root FSDP model'):
        with strategy.block_backward_sync():
            pass
