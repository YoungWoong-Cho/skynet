# Skynet 앱으로 노트의 미실행 실험 돌리기

- 기준: 2026-10-05. 근거는 작업 트리 코드, 중앙 DB 읽기 전용 조회, sky1 읽기 전용 확인이다.
- 대상: owner와, 노트에 적힌 실험을 이어서 돌릴 Claude 세션
- 표기:
  - `(추정)`: 직접 확인하지 않은 판단
  - `path:line`: 근거 위치
  - UI 라벨: `static/index.html`과 `static/app.js`가 기준이다. README/docs와 다르면 코드를 따른다(부록 A).
- 원본/Skynet 구분: 모든 절에서 **원본**(논문·공식 구현)과 **Skynet**(Skynet에서 정한 설정·변경)을 나눠 쓴다(AGENTS.md).

---

## 1. 한눈에 보기

### 1-1. 전체 흐름

| # | 단계 | 어디서 | 필요한 것 |
|---|---|---|---|
| 1 | 외부 코드를 40자 commit으로 고정 | Experiments → Submit → **1. Algorithm and code** | 브라우저. 학습 job이 그 commit을 직접 clone한다. Commit은 고른 **Branch**의 최신 50개 commit 중에서만 고를 수 있다(§5-1) |
| 2 | Python 환경 준비. 공유 checkout은 profile 검증, Convert, evaluator에 쓸 때만 | 클러스터 `paths.environments`, `paths.shared_repositories` | 클러스터 작업(owner) |
| 3 | runtime profile 선언 | `config/clusters/skynet.json` `runtime_profiles` | 설정 변경과 앱 재시작 |
| 4 | 어댑터 | Experiments → **Adapters**(선언형), 또는 `skynet_app/adapters/<x>_manifest.py`(built-in) | 선언형은 브라우저만. built-in은 코드 변경과 앱 재시작 |
| 5 | 데이터 | Data → **Collect** / **Recordings** → **Convert**, 또는 Data → **Datasets** → **New**(HF import 또는 Add files) | 브라우저. 학습 데이터는 Datasets 탭에 등록한 것만 고를 수 있다(§4-1) |
| 6 | 실험 정의와 제출 | Experiments → **Submit**(또는 **Presets**) → **Preview sbatch** → **Create and submit** | 브라우저 |
| 7 | 모니터링 | Experiments → **Runs** → **View** | 브라우저 |
| 8 | 평가 | Runs → **View** → **Start evaluation**, 또는 Evaluations → **Submit** | 브라우저. 평가 계획이 지정한 evaluator profile만 `runtime verified`여야 한다(recorded DexVerse suite는 `isaacsim-5.1.0_isaaclab-2.3.2_py311`). 학습 profile은 해당 없음. attestation은 CLI로 Slurm에 제출하므로 owner가 한다(§2-4) |
| 9 | 결과 기록 | Experiments → **Notes**, 또는 `/api/notes` | 브라우저 또는 API |

### 1-2. 어떤 경로를 쓰나

| 외부 코드의 성격 | 경로 | 근거 |
|---|---|---|
| 한 번 돌릴 스크립트(전처리, rollout, 분석) 또는 자체 데이터를 쓰는 trainer | built-in `generic` 어댑터 "Custom structured command"에 argv를 넣는다 | `adapters/__init__.py:479-505`. Lightwheel이 이미 이 방식이다 |
| 공개 trainer이고 CLI가 있으며, Skynet 시뮬레이터 평가는 필요 없음 | 브라우저 선언형 어댑터 | §3-2 |
| Skynet recording을 Convert해서 학습하거나 `dexverse_recorded`로 평가 | built-in 어댑터(HAT/DP 방식) | Convert 대상 여부는 manifest의 `recording_conversion` 선언이 정한다(`dataset_formats.py:12-43`, slug 검사 없음). 실제로 코드·설정 변경이 필요한 곳: `skynet.json`의 loader 검증 runtime profile(`policy_exports_cluster.py:52-68`), capsule에 넣는 loader 코드(`policy_exports.py:385-391`), recorded 평가 등록(`evaluation_compatibility.py:16-24`), HAT 전용 slug hook(`evaluation_targets.py:9-13`, `recording_sampling.py:42-46`) |

### 1-3. 처음 한 번 할 일

1. 이메일을 넣고 **Open workspace**를 누른다(`docs/email-workspaces.md:3`).
2. **Settings → Storage** 탭 → **Cluster storage**에서 **Base path**(필수, 절대 경로, placeholder `/coc/flash7/yourname`, "Choose a new or empty cluster directory…")를 넣고 **Validate and initialize**를 누른다(`index.html:4576-4630`).
   - 새 run의 디렉터리, 로그, job source cache는 이 base path(`work_root`) 아래에 생긴다(`workspace_storage.py:17-48`, `slurm.py:1406-1407`). 이 가이드의 `{work_root}`, `{paths.*}`는 이 값을 뜻한다.
   - 지금 `workspace_storage`에는 owner `legacy` → `/coc/flash7/ycho420` 하나만 있다. `human-policy-hat` run 211개(hat-main 210개 전부 포함)는 그 전 root인 `/coc/flash2/ycho420/skynet` 아래에 있다. 기존 run의 위치는 항상 그 run에 기록된 `runs.run_directory`를 본다(§6-1).
3. 헤더의 **Cluster gateway**(`index.html:192-195`)를 고른다. 보이는 선택지는 **Auto: sky1, then sky2**, **Prefer sky1**, **Prefer sky2**다(값 `auto`/`sky1`/`sky2`, 서버가 `CLUSTER.gateways`로 채운다: `main.py:477-479`). 이름을 고른 것은 우선순위일 뿐이다. 모든 제출 요청에 실린다.
4. W&B를 쓰면 **Settings → Connections → Weights & Biases**에서 **API key**와 **Entity (optional)**를 넣고 **Connect**를 누른다(`index.html:4423-4477`). 연결 없이 W&B를 켜면 preview와 submit이 모두 실패한다(`pipeline_api.py:6123-6140`).
5. 알림이 필요하면 **Settings → Notifications → Slack**에서 webhook URL을 넣고 **Connect**를 누른다.

### 1-4. 재시작 규칙

- 지금 서버는 `--reload` 없이 떠 있다: `.venv/bin/python -m uvicorn skynet_app.main:app --host 0.0.0.0 --port 8080 ... --env-file .../data/operator.env`.
  - 따라서 코드나 `config/` 변경은 모두 앱을 재시작해야 반영된다.
  - README의 개발 명령(`--reload-dir skynet_app --reload-dir ops`)도 `config/`는 감시하지 않는다.
- `skynet.json`은 import 시점에 `lru_cache`로 읽힌다(`cluster_config.py:394-404`). 이 파일을 고치는 API는 일부러 만들지 않았다(README:187).
- 시작 시 환경변수가 `skynet.json`을 덮어쓸 수 있다(`cluster_config.py:351-391`): `SKYNET_CLUSTER_CONFIG`(다른 profile 파일), `SKYNET_SSH_HOSTS`(gateways 대체), `SKYNET_HOME_ROOT`/`SKYNET_WORK_ROOT`(경로 rebase), `SKYNET_SLURM_BIN`, `SKYNET_GPU_USAGE_COMMAND`, `SKYNET_GPU_USAGE_INTERPRETER`, `SKYNET_QUEUE_<POLICY>_{PARTITION,ACCOUNT,MAX_TIME_SECONDS}`. 지금 `data/operator.env`에는 `OMNI_KIT_ACCEPT_EULA`만 있으므로 이 가이드의 값이 그대로 적용된다.

---

## 2. 외부 repo와 코드 가져오기

### 2-1. checkout은 역할이 다른 세 종류다

| 종류 | 경로 | 누가 만드나 | 쓰는 곳 |
|---|---|---|---|
| (a) 학습 job의 source cache | `{work_root}/repos/{name}-{sha256(repo_url)[:8]}/{40자 commit}`(`slurm.py:1152-1164`) | job이 compute node에서 직접 만든다 | `SKYNET_SOURCE_DIR`, `SKYNET_PROJECT_DIR`. 학습 job(generic 포함)의 코드는 항상 이 경로에서 나온다. argv는 `cwd=project_dir`에서 돈다(`slurm.py:961`) |
| (b) 운영자가 고정한 checkout | 보통 `{paths.shared_repositories}/<Name>/<sha 또는 tag>`. profile이 (a) 경로를 가리켜도 된다(EgoVerse가 그렇다, 아래) | owner(또는 앞선 job) | runtime profile의 `source_prerequisites`만 참조한다. 학습에는 안 쓰고, 아래 네 곳에서만 쓴다: Convert의 loader 검증, evaluator `source_dir`, readiness CLI, profile 검증 probe(`source_prerequisites`를 읽는 코드는 `cluster_config.py`, `experiments.py`, `pipeline_api.py`, `policy_exports_cluster.py`, `runtime_readiness.py`뿐이고 `slurm.py`에는 없다) |
| (c) 디렉터리 prerequisite | 예: `/coc/flash7/ycho420/repos/skynet-dexverse/30cc673e…` (`kind: "directory"`) | owner | Isaac evaluator profile |

(a) source cache 규칙:

- 디렉터리 이름은 `_safe_identifier(URL 마지막 조각에서 .git을 뺀 것, 최대 48자)-<URL 문자열 전체의 sha256 앞 8자>`다(`slurm.py:1052-1056, 1155-1159`).
  - URL 철자가 다르면 cache도 따로 생긴다. 예(계산값): `https://github.com/RogerQi/human-policy` → `human-policy-217c2d08`, `…/human-policy.git` → `human-policy-90784637`, `…/human-policy/` → `human-policy-2e62ff0b`.
  - 그래서 Repository에는 앞선 run과 정확히 같은 `https://` 철자(`.git` 유무, 끝 `/` 포함)를 쓴다.
- job은 `flock` 아래에서 아래 순서로 clone한다(clone 블록 `slurm.py:1606-1638`).
  1. `git clone --no-checkout -- <입력한 URL 문자열 그대로>`(`slurm.py:1621`)
  2. `fetch <sha>`
  3. `checkout --detach`
  4. submodule 갱신. `source.include_submodules`가 켜졌을 때만(기본 켜짐, `experiments.py:72-73`). submodule의 GitHub SSH URL만 HTTPS로 바꾼다(`GITHUB_SSH_TO_HTTPS`, `slurm.py:1595-1597`)
  5. `git lfs pull`. `source.include_git_lfs`가 켜졌고(기본 켜짐) node에 `git-lfs`가 있을 때만
- 폼은 `https://`, `ssh://`, `git://`, `git@host:owner/repo`를 모두 받고 문자열을 바꾸지 않는다(`source_control.py:1016-1039`). 하지만 코드 주석은 "compute nodes carry no GitHub SSH identity"라고 적는다(`slurm.py:1592-1594`). 따라서 SSH URL은 compute node clone에서 실패한다(추정, 주석 근거). 항상 `https://` URL을 쓴다.
  - private repo를 쓰려면 클러스터에 HTTPS credential이 있어야 한다(owner).
- 종료 코드:
  - exit 65: 기존 cache의 `origin`이 입력한 URL과 다르거나(`slurm.py:1623-1627`), checkout 뒤 HEAD가 고른 commit과 다르다(`slurm.py:1631`). 디렉터리 이름에 URL 해시가 들어가므로 origin 불일치는 사실상 손으로 만들거나 고친 cache에서만 생긴다.
  - exit 66: **Project subdirectory**가 그 commit에 없다(`slurm.py:1636`).
  - 따라서 (a) 경로를 손으로 만들거나 고치지 않는다.
- sky1에서 확인한 예(지금 legacy root): `/coc/flash7/ycho420/repos/human-policy-217c2d08/2d9d73cc…`(옆에 `.lock`), `diffusion_policy-1d5db43e/5ba07ac6…`, `EgoVerse-f7f08862/e17cf98f…`, `XPolicyLab-aceb81ca/9c98a3aa…`. HAT 본 실험 run의 cache는 그 전 root인 `/coc/flash2/ycho420/skynet/repos/human-policy-217c2d08/2d9d73cc…`에 있다.

(b) 운영자 checkout 규칙:

- sky1의 `repos/shared`에는 `diffusion_policy/5ba07ac…`, `human-policy/2d9d73cc…`, `IsaacLab/{v2.2.0,v2.3.2}`, `IsaacLabEvalTasks/460f287…`, `XPolicyLab/9c98a3a…`가 있다.
- 설정 키 `paths.shared_repositories`(`skynet.json:27`)는 이름 규칙일 뿐이다. 이 키를 읽는 코드는 없다.
- `egoverse-native`와 `egoverse-pi`는 `repos/shared`가 아니라 (a) cache 경로 `/coc/flash7/ycho420/repos/EgoVerse-f7f08862/e17cf98…`를 `git_checkout` prerequisite로 쓴다(`skynet.json:295-302, 358-371`).
  - `egoverse-native`의 `environment_path`는 `<그 cache>/.venv`다. sky1의 `pyvenv.cfg`는 `uv = 0.8.14`, `prompt = egomimic`이고 checkout에 `uv.lock`이 있다. 즉 앞선 uv job이 만든 환경을 `existing` 환경으로 다시 쓴다.
  - 따라서 이 cache 디렉터리는 지우지 않는다.
- Convert의 loader 검증은 profile의 **첫** prerequisite 경로를 `{repository}`로, `environment_path/bin/python`을 Python으로 쓴다(`policy_exports_cluster.py:52-66`).
- evaluator `source_dir`은 `revision`이 suite의 `task_catalog_provenance.revision`과 같은 prerequisite다(`pipeline_api.py:9541-9566`).

### 2-2. 클러스터 준비 (owner, 클러스터 쓰기)

현재 repo와 docs에는 checkout과 환경을 만드는 절차가 없다. git 이력의 `ops/runtimes/`는 UniDex 전용이었다. 들어 있던 것은 `bootstrap_unidex.py`(`uv venv --python <기존 python3.11> <새 prefix>` 뒤 `uv pip install --require-hashes -r <lock>`), `unidex.md`, lock 파일뿐이었고, `12234f0`에서 UniDex와 함께 지워졌다. 공유 checkout을 만드는 절차는 이력에도 없다(`git log --all -S 'repos/shared'`는 이 문서 커밋만 찾는다). 아래 명령은 sky1의 기존 checkout(`.git/logs/HEAD`, shallow 여부, `remote.origin.partialclonefilter`)과 환경(`conda-meta/history`, `pyvenv.cfg`, `*.dist-info/INSTALLER`, `direct_url.json`)을 읽기 전용으로 확인해 맞춘 것이다(2026-10-06).

```bash
# 1) 공유 checkout: 디렉터리 하나에 commit(또는 tag) 하나. profile 검증, Convert loader 검증, evaluator source_dir, readiness CLI가 쓴다
#    IsaacLab, IsaacLabEvalTasks checkout은 conda 환경의 editable 설치 원본이기도 하다(2-a). 지우면 그 환경과 그 위 overlay(skynet-dp-py311)에서 import isaaclab이 깨진다
# 1-a) commit 고정: diffusion_policy, human-policy, IsaacLabEvalTasks, XPolicyLab. reflog는 "clone: from <url>" → "checkout: moving from main to <sha>"다(--no-checkout을 썼어도 reflog는 같다)
git clone <url> /coc/flash7/ycho420/repos/shared/<Name>/<sha>        # XPolicyLab만 --filter=blob:none(partial clone)
git -C /coc/flash7/ycho420/repos/shared/<Name>/<sha> checkout <sha>    # detached HEAD
# 1-b) tag 고정: IsaacLab/v2.2.0, IsaacLab/v2.3.2. reflog에는 clone 한 줄뿐이고 HEAD는 tag에 detached다. profile revision도 tag다(skynet.json:62, 160)
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git /coc/flash7/ycho420/repos/shared/IsaacLab/v2.3.2   # v2.2.0은 --depth 1(shallow)
# submodule: 기존 human-policy(human_data/opentv)와 IsaacLabEvalTasks(submodules/Isaac-GR00T)는 초기화하지 않았다. 필요할 때만
git -C <checkout> submodule update --init --recursive
# 확인: profile 검증은 rev-parse HEAD와 rev-parse <revision>^{commit}이 같은지만 본다(pipeline_api.py:3090-3100, 3180-3189). 작업 트리 변경은 보지 않는다(XPolicyLab에는 수정된 파일이 1개 있다)
git -C <checkout> rev-parse HEAD
git -C <checkout> rev-parse '<revision>^{commit}'

# 2-a) conda prefix 환경: isaacsim-5.1.0_isaaclab-2.3.2_py311, groot-isaacsim-5.0.0_isaaclab-2.2.0_py311(dexverse는 conda create -n dexverse python=3.11)
#      bin/activate가 없으므로 profile backend는 conda다(conda run -p, slurm.py:1358)
/nethome/ycho420/miniconda3/bin/conda create -y -p /coc/flash7/ycho420/envs/<name> python=3.11 pip    # groot는 python=3.11만
/nethome/ycho420/miniconda3/bin/conda install --prefix /coc/flash7/ycho420/envs/<name> --yes cmake                          # isaacsim-5.1.0만
/nethome/ycho420/miniconda3/bin/conda install --prefix /coc/flash7/ycho420/envs/<name> --yes --channel conda-forge libglu    # isaacsim-5.1.0만
/coc/flash7/ycho420/envs/<name>/bin/python -m pip install <버전을 고정한 패키지>                 # isaacsim 5.1.0.0/5.0.0.0 wheel 등. INSTALLER=pip
/coc/flash7/ycho420/envs/<name>/bin/python -m pip install -e <공유 IsaacLab>/source/<패키지>    # direct_url.json이 editable이다. isaaclab.sh --install을 썼는지는 구별할 수 없다
# 2-b) venv overlay: skynet-dp-py311, skynet-egoverse-pi-981483dc. profile backend는 existing이다(bin/activate 필요, slurm.py:1370)
<base>/bin/python -m venv --system-site-packages /coc/flash7/ycho420/envs/<name>
/coc/flash7/ycho420/envs/<name>/bin/python -m pip install <버전을 고정한 패키지>
#      skynet-egoverse-pi는 base가 uv venv라서 --system-site-packages로는 uv cpython 3.11.13만 보인다. EgoVerse 패키지는
#      site-packages/egoverse-native.pth(EgoVerse .venv site-packages, checkout, external/openpi/src, openpi-client/src 네 줄)로 붙였다
```

- 기존 환경이 실제로 만들어진 방식(`pyvenv.cfg`, 읽기 전용 확인):
  - `envs/skynet-dp-py311`: `isaacsim-5.1.0_isaaclab-2.3.2_py311/bin/python -m venv --system-site-packages`(Python 3.11.16). HAT, DP, `xpolicylab-act`가 이 환경을 함께 쓴다.
  - `envs/skynet-egoverse-pi-981483dc`: `EgoVerse-f7f08862/e17cf98…/.venv/bin/python`에서 `venv --system-site-packages`
- 다른 방법: `uv` backend를 쓰고 job이 환경을 만든다(§2-3).
  - 자동 감지는 `uv.lock`과 `pyproject.toml`이 함께 있어야 한다(README:300). `uv run --frozen`이므로 `uv.lock`은 반드시 있어야 한다.
  - job은 `uv run --frozen --project $SKYNET_PROJECT_DIR`(`slurm.py:1325-1327`)로 돌고, 환경은 `{work_root}/repos/<name>-<hash8>/<sha>/<subdir>/.venv`에 남는다(EgoVerse `.venv`가 그 예. uv 기본 동작에 기댄 판단).
  - uv는 찾을 수 있는 곳에 있거나, profile에 `bootstrap_uv: true`가 있어야 한다(§2-3 표).
- sky1 `/coc/flash7/ycho420/envs`에 있는 것: `dexverse`, `groot-isaacsim-5.0.0_isaaclab-2.2.0_py311`, `isaacsim-5.1.0_isaaclab-2.3.2_py311`, `project-overlays`, `skynet-dp-py311`, `skynet-egoverse-pi-981483dc`, `skynet-gpu-usage`, `system-libs`, `warp-retargeting`

### 2-3. `config/clusters/skynet.json`에 runtime profile 선언

위치는 `runtime_profiles`(line 43)다. 지금 profile은 7개다.

| Profile | line |
|---|---|
| `isaacsim-5.1.0_isaaclab-2.3.2_py311` | 44 |
| `groot-isaacsim-5.0.0_isaaclab-2.2.0_py311` | 136 |
| `xpolicylab-act` | 263 |
| `egoverse-native` | 291 |
| `egoverse-pi` | 353 |
| `human-policy-hat` | 427 |
| `diffusion-policy` | 455 |

HAT profile(`skynet.json:427-453`)을 본뜬 학습용 틀이다. 학습 profile에는 attestation이나 `compute_smoke`가 필요 없다. evaluator profile로 쓰려면 §2-4의 요건을 더 채워야 한다.

```json
"<profile-id>": {
  "label": "<Model> · <official source> (<runtime 설명>)",
  "description": "Official <repo> pinned source; Skynet runtime reuse is distinct from upstream installation instructions.",
  "backend": "existing",
  "environment_path": "/coc/flash7/ycho420/envs/<name>",
  "source_prerequisites": [
    {"kind": "git_checkout", "name": "<Official Name>",
     "path": "/coc/flash7/ycho420/repos/shared/<Name>/<sha>", "revision": "<sha>", "required": true}
  ],
  "verification": {
    "python_version": "3.11",
    "distributions": {"torch": "2.7.0+cu128"},
    "python_imports": ["torch"]
  }
}
```

스키마 규칙(`skynet_app/cluster_config.py`):

- profile id는 `[a-z0-9][a-z0-9._-]{0,127}` 형식이다(306-308).
- 모르는 필드가 있으면 실패한다(`extra="forbid"`, 16).
- `RuntimeProfileConfig`(165-209):
  - `label`은 필수이고 160자 이하다.
  - `backend`는 `uv | conda | apptainer | existing` 중 하나다(168).
  - `environment_path`는 절대 경로다. `conda`이면 필수다(201).
  - `container_image`는 `apptainer`이면 필수다(203).
  - 그 밖의 필드: `environment_operations`, `lock_file`, `uv_executable`, `bootstrap_uv`, `environment`, `versions`
  - `environment_operations`(예: `LD_LIBRARY_PATH` prepend)는 evaluator, readiness, collection job에서만 적용된다(`runtime_readiness.py:231,472`, `adapters/policy_simulator.py:36`, `adapters/groot_isaaclab_bridge.py:968`, `collection.py:1379`). 학습 job에는 profile의 `environment`와 `training_environment`만 들어간다(`pipeline_api.py:2734-2737`). `slurm.py`는 이 필드를 읽지 않는다.
- `RuntimeSourcePrerequisite`(48-69):
  - `kind`는 `git_checkout | directory`다.
  - `path`는 절대 경로다.
  - `git_checkout`이면 `revision`이 필수다(66-69).
- `RuntimeProfileVerification`(118-144): `python_version`, `distributions`(버전 prefix), `required_distributions`, `python_imports`, `executables`, `shared_libraries`, `gym_registrations`, `glibc_minimum`, `requires_gpu`, `compute_attestation_path`(절대 경로), `compute_smoke`, `pip_check` 등

backend별 실행 방식(`slurm.py:1289-1375`):

| backend | 실행 |
|---|---|
| `existing` | `[[ -r <env>/bin/activate ]]`를 먼저 보고 없으면 exit 69("preflight: existing environment activation script missing"). 그다음 `source <env>/bin/activate`, `python3 runtime-wrapper.py`(`slurm.py:1366-1374`). 따라서 venv 형태(`bin/activate`가 있는) prefix만 된다. 경로가 없으면 node의 시스템 `python3`가 돈다 |
| `conda` | `conda run -p <env>`. `lock_file`이 있으면 `conda-lock install`. conda prefix(예: sky1 `envs/isaacsim-5.1.0_isaaclab-2.3.2_py311`에는 `bin/activate`가 없다)는 이 backend를 쓴다 |
| `apptainer` | `apptainer exec --nv <image>` |
| `uv` | `uv run --frozen --project $SKYNET_PROJECT_DIR`. repo에 `uv.lock`이 있어야 한다. uv는 `runtime.uv_executable`, `$WORK_ROOT/.local/bin/uv`, `$HOME/.local/bin/uv`, `PATH`, `python3 -m uv` 순으로 찾는다(`slurm.py:1295-1302`). 못 찾으면 profile에 `bootstrap_uv: true`일 때만 `$WORK_ROOT/.cache/uv/bootstrap-<runtime.uv_version>`에 고정 버전(기본 0.8.14)을 설치한다(1304-1314). 아니면 exit 69 |

profile이 실험에 들어가는 방식(`pipeline_api.py:2648-2800`):

- `source.revision`은 40자 sha여야 한다: "preview and submission require a pinned 40-character commit"(2671-2676).
- 고른 profile의 backend는 manifest `runtime.allowed_backends` 안에 있어야 한다(2689-2704).
- profile의 `environment_path`, `container_image`, `lock_file`, `uv_executable`, `bootstrap_uv`가 요청 값을 대신한다.
- `profile_snapshot`과 `profile_snapshot_sha256`이 spec에 박힌다(2709-2744).
- `CLUSTER.training_environment`(`NCCL_P2P_DISABLE=1`, `skynet.json:544`)가 runtime 환경에 합쳐진다.

코드에 박힌 profile id와 queue key(이름을 바꾸거나 지우지 않는다):

- Convert worker는 항상 `xpolicylab-act` profile의 환경으로(`policy_exports_cluster.py:99-100`), queue key `normal`에서 돈다(73).
- observation 준비는 `isaacsim-5.1.0_isaaclab-2.3.2_py311`(render) 또는 `xpolicylab-act`를 쓴다(`observation_preparation.py:233-236`, queue `normal`).
- recorded 평가 evaluator는 `isaacsim-5.1.0_isaaclab-2.3.2_py311`로 고정이다(`evaluation_compatibility.py:186`).
- Convert의 loader 검증은 어댑터 `loader_validation.runtime_profile`의 환경과 그 profile의 **첫** `source_prerequisite`를 쓴다(`policy_exports_cluster.py:52-68`). 새 built-in 어댑터의 이 profile은 `skynet.json`에 있어야 하고, 첫 prerequisite는 upstream checkout이어야 한다.
- 따라서 `xpolicylab-act`, `isaacsim-5.1.0_isaaclab-2.3.2_py311`, queue key `normal`을 바꾸면 Convert, observation 준비, recorded 평가가 깨진다.

### 2-4. 검증

동작(`_probe_runtime_profile`, `pipeline_api.py:2990-3247`):

- `conda`/`existing`이고 `compute_attestation_path`가 없으면 바로 `configured` 상태를 낸다. 메시지는 "has no compute_attestation_path; configure one and run the one-GPU evaluator readiness smoke"다(3053-3063).
  - 지금 `human-policy-hat`, `diffusion-policy`, `xpolicylab-act`가 여기에 해당한다.
  - 이때는 환경도 prerequisite도 검사하지 않는다. 학습 전용 profile은 제출 전에 아무것도 확인되지 않으므로, 잘못된 경로는 job이 돌 때(exit 69 등) 드러난다. owner가 먼저 `ssh sky1 'test -x <env>/bin/python && test -r <env>/bin/activate'`와 checkout `git -C <path> rev-parse HEAD`를 확인한다.
- attestation 경로가 있으면 SSH 한 번(30 s 제한)으로 아래를 본다(3064-3184).
  - 환경 디렉터리와 `bin/python`
  - attestation JSON(schema `skynet.runtime-readiness/v1`)
  - prerequisite마다 `test -d`, 그리고 `git_checkout`이면 `git rev-parse HEAD`와 `<revision>^{commit}`이 같은지
- attestation은 현재 profile의 `profile_snapshot_sha256`과 suite의 `suite_contract_sha256`이 기록값과 같아야 유효하다. 다르면 "compute readiness attestation is stale for the configured runtime profile; rerun…"으로 거부된다(`pipeline_api.py:2825-2841`, 기록은 `runtime_readiness.py:803-807`).
  - `skynet.json`에서 그 profile의 JSON을 조금이라도 고치면(label, description 포함) snapshot hash가 바뀌어 attestation이 stale이 된다. readiness smoke를 다시 돌려야 한다.

막는 범위:

- **학습 제출은 `configured`로도 된다.** 학습 profile에는 attestation도 `compute_smoke`도 필요 없다. HAT 평가가 성공한 것도 학습 profile `human-policy-hat`(`configured`)이 검사 대상이 아니기 때문이다.
- **평가 계획이 `evaluation_runtime_profile_id`로 지정한 evaluator profile만 `runtime verified`여야 한다**(`pipeline_api.py:9933-9984`). 아니면 "evaluation runtime profile X is not ready: …"(`pipeline_api.py:9617-9677`).
  - recorded DexVerse suite(`dexverse_recorded`, `dexverse_training_episode`): `isaacsim-5.1.0_isaaclab-2.3.2_py311`(`evaluation_compatibility.py:186`)
  - GR00T `groot_gr1_isaaclab_evaltasks`: `groot-isaacsim-5.0.0_isaaclab-2.2.0_py311`(`adapters/__init__.py:3087`)
  - EgoVerse `egoverse_held_out`: `egoverse-pi` 또는 `egoverse-native`(`adapters/egoverse_manifest.py:289-291`)
- evaluator attestation은 CLI로만 만든다. Slurm 제출이므로 owner가 한다(`skynet_app/runtime_readiness.py:828-866`). repo 루트에서 실행한다.
  ```bash
  # Isaac profile이면 같은 셸에서 EULA를 직접 수락한다. data/operator.env는 서버만 읽는다.
  export OMNI_KIT_ACCEPT_EULA=YES
  .venv/bin/python -m skynet_app.runtime_readiness render-sbatch --profile <id> --suite <compute_smoke.suite>   # 스크립트만 출력
  .venv/bin/python -m skynet_app.runtime_readiness submit-sbatch --profile <id> --suite <compute_smoke.suite> \
      --run-id <stable-id> [--gateway auto] [--gpu-type ... --queue-policy ... --memory-gb ... --time-limit ... --node ...]
  ```
  - `--run-id`와 `--gateway`(기본 `auto`)는 `submit-sbatch`에만 있다. 같은 `--run-id`로 다시 내면 중복 없이 복구한다.
  - CPU는 `--cpus-per-task`와 무관하게 GPU당 8개다(`runtime_readiness.py:285-286`).
  - Isaac suite는 job 안에서 `OMNI_KIT_ACCEPT_EULA`가 `YES`가 아니면 exit 2로 끝난다(`runtime_readiness.py:357-360`). CLI 프로세스 자신의 환경에서 이 변수 하나만 넘긴다(`cluster_runtime.py:105-115`).
- evaluator profile 요건(`build_readiness_contract`, `runtime_readiness.py:133-191`):
  - `verification.compute_attestation_path`
  - `verification.compute_smoke`: `suite`(= `--suite`), `adapter`, `capsule_asset`(built-in 어댑터 evaluation command의 `capsule_files`에 있어야 한다), `capsule_sha256`(그 파일과 일치), `argv`, `resources`, `result_schema`, `required_checks`. 없으면 "runtime profile X does not declare compute_smoke"
  - `verification.gym_registrations`의 id가 suite task environment id와 정확히 같아야 한다.
  - `revision`이 suite `task_catalog_provenance.revision`과 같은 prerequisite가 있어야 한다.
  - §2-3의 HAT 틀에는 이 항목들이 없다. `human-policy-hat`으로 `render-sbatch`를 하면 compute_smoke가 없다는 오류가 난다.
- 이미 있는 attestation과 지금 상태(현재 snapshot hash와 sky1 파일의 기록값을 비교. 실시간 probe가 아니므로 추정):

  | profile | attestation | 상태 |
  |---|---|---|
  | `isaacsim-5.1.0_isaaclab-2.3.2_py311` | `/coc/flash7/ycho420/envs/isaacsim-5.1.0_isaaclab-2.3.2_py311/.skynet/compute-readiness-v1.json` | 유효(`7cfe3dba…` 일치) |
  | `groot-isaacsim-5.0.0_isaaclab-2.2.0_py311` | `/coc/flash7/ycho420/envs/groot-isaacsim-5.0.0_isaaclab-2.2.0_py311/.skynet/compute-readiness-v1.json` | stale(현재 `54066ee5…`, 기록 `549db5b8…`) |
  | `egoverse-native` | `/coc/flash7/ycho420/jobs/runtime-readiness/egoverse-native.json` | stale(현재 `c8afcf28…`, 기록 `98dde0c1…`) |
  | `egoverse-pi` | `/coc/flash7/ycho420/jobs/runtime-readiness/egoverse-pi.json` | stale(현재 `ff8c1d87…`, 기록 `45f7c4b8…`) |

  - stale인 셋은 **Named runtime profile**에 `"<label> / configured"`로 보인다(probe를 돈 응답이든 아니든 같다). probe는 기록값이 현재 snapshot hash와 다르면 "compute readiness attestation is stale for the configured runtime profile; rerun the one-GPU evaluator readiness smoke" 오류를 넣는다(`pipeline_api.py:2827-2833`). 오류가 하나라도 있으면 `status`는 `configured`, `runtime_verified`는 `false`가 된다(3192-3205). UI는 `runtime_verified`만 보고 상태를 정한다(`app.js:6234`, 옵션 문구 6424, 도움말 6491, 상태 줄 6531).
    - stale을 따로 표시하지 않는다. attestation 파일이 없을 때, `compute_attestation_path`가 없을 때(`xpolicylab-act`, `human-policy-hat`, `diffusion-policy`, 3054-3063), SSH probe가 실패할 때(3207-3212)도 똑같이 `configured`다. 옵션은 막히지 않고, 학습 제출도 `runtime_verified`를 보지 않는다.
    - 반대로 `runtime verified`는 probe를 돈 응답에서만 나온다. `/api/source/inspect`가 DB cache를 쓰면 `runtime_profiles(gateway)`를 verify 없이 붙이므로(`pipeline_api.py:10948-10950`, `cluster_config.py:336-345`) 유효한 `isaacsim-5.1.0_isaaclab-2.3.2_py311`도 `configured`로 보인다. 그 commit을 처음 검사할 때와 **Refresh refs**(→ `refresh=true`, `app.js:20363-20368, 1397-1399`)를 눌렀을 때만 probe한다(`pipeline_api.py:10971`).
    - 평가 생성은 suite 계약까지 넣어 `refresh`로 다시 probe한다. `runtime_verified`가 아니면 "evaluation runtime profile X is not ready: …"로 막는다(`pipeline_api.py:9617-9676, 10316-10318`). 따라서 GR00T IsaacLab 평가와 EgoVerse 평가를 하려면 smoke를 다시 돌려야 한다. 2026-10-06에 sky1 기록값을 다시 읽었을 때도 세 profile은 stale이었다.

확인하는 곳:

- 상태: Experiments → Submit → **3. Runtime** → **Named runtime profile**에 `"<label> / runtime verified"` 또는 `"/ configured"`로 보인다(`app.js:6420-6429`).
- 오류 문자열은 UI에 나오지 않는다. `app.js:6236`이 `verification`을 받아 두지만 아무 데도 그리지 않는다. 옵션 문구(6424-6427)와 도움말(6491)은 상태, id, 버전만 보여 준다.
  - 오류는 `GET /api/source/inspect?repo_url=…&revision=…&refresh=true` 응답의 `runtime_profiles[].verification.errors`에서 본다.
  - `refresh=true`를 붙여야 cache를 비우고 `runtime_profiles(gateway, verify=True)`로 다시 SSH probe한다(`pipeline_api.py:10873, 10924-10927, 10970`). profile, 환경, attestation을 바꾼 뒤에는 꼭 붙인다.
  - 모든 `/api` route는 열린 email-workspace session이 필요하다(`require_workspace_records`, `pipeline_api.py:10409`). 세션 없는 curl은 안 된다. 브라우저 devtools의 Network 탭에서 응답을 읽는다.

### 2-5. profile 없이 브라우저만으로

- **3. Runtime**에서 Runtime을 **Existing environment (manual)**로 고르고, "Custom absolute prefix, lock path, or image reference"에 환경 경로를 넣는다(`app.js:7160-7176`).
  - `existing`은 `bin/activate`가 있는 venv 형태 prefix여야 한다. conda prefix는 **Conda (manual)**을 고른다(§2-3 표).
- manual backend 중에서 경로 칸이 비어도 되는 것은 `existing`뿐이다(`app.js:6346-6367`). 비우면 node의 시스템 `python3`가 돈다.
- 제출 전에는 이 경로를 아무도 검사하지 않는다. owner가 먼저 `ssh sky1 'test -x <env>/bin/python'`으로 확인한다.
- 이렇게 돌린 run에는 profile snapshot과 검증 기록이 남지 않는다.
  - 재현 기록이 필요한 실험은 profile을 쓴다.

### 2-6. 원본과 Skynet 구분 규칙

- profile `description`에 공식 source와 Skynet runtime 재사용을 나눠 쓴다. HAT profile이 예다: "Official HAT source; reuses the existing ACT Python environment. Skynet runtime reuse is distinct from upstream installation instructions."
- 어댑터 `description`과 input field `help`에도 원본 기본값과 Skynet 설정을 나눠 쓴다. 예: "Skynet setting; upstream default is <N>."
- 노트에서는 `**원본 모델/공식 구현:**`과 `**Skynet 별도 구현/변경:**` 문단을 따로 둔다(HAT 노트 §1이 예).

---

## 3. 어댑터

### 3-1. 지금 쓸 수 있는 어댑터 (DB, 보관되지 않은 최신 버전)

| slug | 이름/버전 | 비고 |
|---|---|---|
| `generic` | Custom structured command v145 | 어떤 repo든 argv로 돌린다. run 18개(최근 Lightwheel) |
| `human-policy-hat` | v12 | HAT 연구는 v9로 고정했다. run 214개 |
| `diffusion-policy` | v3 | run 3개 |
| `egoverse-act`, `egoverse-hpt`, `egoverse-pi` | | `egoverse-hpt`는 EgoVerse `e17cf98f`에 고정되어 있고 RGB/joint만 받는다(`egoverse_manifest.py:12-13`) |
| `openpi`, `groot`, `dexmimicgen`, `get_zero` | | `legacy_handler` 클래스 방식 |
| `xpolicylab-act` + `xpolicylab-*-native` 37개 | | |

- `egoverse-hpt-*`, `egoverse-pi05-*`, `dexverse` 계열은 보관(archived) 상태다.
- DB의 adapter row는 71개이고 사용자가 작성한 버전은 0개다. 모두 `__seed__` 또는 `__migration__`이다.

### 3-2. 선언형 어댑터를 브라우저로 만들기 (재시작 없음)

화면은 Experiments → **Adapters** 탭이다(`index.html:724-733`, 패널 1640-1700).

- 툴바: filter, **Show archived**, **New**
- 행 동작(`app.js:4619-4627`): **View**, **Edit**, **Duplicate**, **Archive**/**Restore**, **Delete**
  - **Edit**은 보관되지 않았고 편집 권한이 있을 때만, **Duplicate**는 보관되지 않은 어댑터에만 나온다.
  - **Archive**/**Restore**와 **Delete**는 편집 권한이 있을 때만 나온다(`adapter.editable === false`이면 View와 Duplicate만 남는다. 예: 다른 email workspace에서 본 built-in).
  - **Archive**는 HTTP `DELETE /api/adapters/{id}`를 쓰지만 삭제가 아니라 보관이다(`app.js:5747-5762`).
  - **Delete**는 maintenance 의존성 미리보기를 거친다: `GET` → `DELETE /api/maintenance/history/adapter/{id}`
  - **Duplicate**는 "Name for the duplicated adapter:"와 "Initial version change note:" 두 질문을 띄운 뒤 `POST /api/adapters/{id}/clone` `{name, version_number, change_note}`를 부르고 복제본을 편집 모드로 연다(`app.js:5715-5742`).
- 편집 창(`index.html:1703-1860`):
  - 입력: **Adapter key**("Immutable after creation"), **Display name**, **JSON manifest**, **Registry description**, **Version change note**, **Repository URL**, **Branch, tag, or commit SHA**
  - 버튼: **Edit**, **Validate**, **Save adapter**
  - **Versions** 표

순서:

1. **New**를 누르면 slug `new-adapter`, backend 네 개, GPU 1개, `supports_resume: false`, 빈 `train`이 채워진다(`app.js:5452-5514`).
2. **Adapter key**와 **Display name**을 넣는다.
   - **Validate**와 **Save adapter** 때 UI가 manifest의 `slug`(Adapter key 소문자), `capabilities.name`(= slug), `display_name`을 이 값으로 덮어쓴다(`normalizedEditorManifest`, `app.js:4751-4764`). JSON에 쓴 세 값은 무시된다.
   - Adapter key는 만든 뒤 바꿀 수 없다. 편집 모드에서는 칸이 잠기고 저장된 slug가 유지된다(`app.js:4793`).
3. manifest를 붙여 넣는다. `AdapterManifest` 검증을 통과하는 틀은 `scratchpad/note-guide/code-adapters/skeleton.json`에 있다. 요점:

```json
{
  "schema_version": "skynet.adapter/v1",
  "slug": "my-policy", "display_name": "My Policy · Official",
  "description": "Original <model> from <repo>@<sha>. Skynet-specific: <bridges/changes>.",
  "runtime": {"allowed_backends": ["existing", "conda"], "recommended_backend": "existing"},
  "capabilities": {"name": "my-policy", "version": 1, "runtime_backends": ["existing", "conda"],
    "supports_multi_gpu_single_node": false, "supports_resume": false,
    "minimum_gpus": 1, "recommended_gpus": 1, "maximum_gpus": 1},
  "defaults": {"hyperparameters": {"learning_rate": 0.0001, "batch_size": 16, "max_steps": 3000, "seed": 42},
               "checkpoint": {"auto_resume": false, "final_selector": "latest"}},
  "train": {
    "argv": ["python", "train.py", "--output", "{{tokens.run_dir}}/artifacts",
             "--dataset", "{{native.config.dataset_path}}", "--batch-size", "{{train.batch.value}}",
             "--lr", "{{train.learning_rate}}", "--max-steps", "{{train.max_steps}}", "--seed", "{{train.seed}}"],
    "supported_canonical_fields": ["train.learning_rate", "train.batch.value", "train.max_steps", "train.seed"],
    "input_fields": [{"path": "native.config.dataset_path", "label": "Training dataset", "kind": "string",
                      "required": true, "data_binding": {"role": "training_data", "formats": ["<format>"],
                      "value_path": "version.path"}}],
    "checkpoint_globs": ["artifacts/checkpoints/*.ckpt"],
    "progress": {"unit": "step", "total_path": "train.max_steps",
                 "source": {"kind": "jsonl", "path": "artifacts/metrics.jsonl",
                            "completed_key": "step", "required_key": "train_loss"}}
  },
  "evaluations": []
}
```

4. **Repository URL**과 **Branch, tag, or commit SHA**를 넣고 **Validate**, 그다음 **Save adapter**를 누른다.
5. Experiments → Submit에서 이 어댑터를 고른다(§5).

manifest 검증 규칙(`adapters/__init__.py:2055-2093`):

- `capabilities.name == slug`
- `runtime.allowed_backends == capabilities.runtime_backends`
- GPU 권장값은 capability 최소/최대 안에 있어야 한다.
- preset id는 서로 달라야 하고, default preset이 실제로 있어야 한다.
- null이 아닌 `defaults.hyperparameters` 값은 `train.supported_canonical_fields`의 경로에 대응해야 한다. `legacy_handler`가 있으면 예외다.

`data_binding.value_path` 고르기(§4-1과 연결). 허용 값은 `version.path`(기본), `mount_path`, `location.path`, `version.manifest_sha256`, `selection`이다(`adapters/__init__.py:1524`).

| value_path | 받는 데이터 |
|---|---|
| `location.path`, `selection` | Convert로 만든 데이터셋만. 검증된 cluster location이 필요하다(`pipeline_api.py:2317-2319`) |
| `version.path`(기본), `contracts` 없음 | Data → **Datasets** 탭에서 등록한(category `dataset`) `READY` 버전도 받는다. **Files** 탭에서 만든 버전(category `file`)은 학습 데이터로 못 쓴다(§4-1) |
| `version.manifest_sha256` | 값으로 manifest SHA를 넘긴다(XPolicyLab Native가 경로와 짝으로 쓴다) |

- `cardinality: "many"`는 `value_path: "selection"`, position 0에서만 된다(`adapters/__init__.py:1530-1534`). 따라서 여러 데이터셋을 받는 binding은 Convert로 만든 데이터셋만 받는다.

- `data_binding` 없는 string 입력으로 경로를 받으면 등록·검증 없이 문자열이 그대로 넘어간다. 제출 때 공통 검사는 required, kind(`str`인지), `minimum`/`maximum`, `choices`(`pipeline_api.py:2414-2441`), `choice_source` 목록(`pipeline_api.py:2518-2549`)뿐이다. 값은 command template에 `str(value)`로 그대로 치환된다(`adapters/__init__.py:2154-2163, 403-410`). 경로가 실제로 있는지, 절대경로인지, 등록된 데이터인지, SHA가 맞는지는 아무도 보지 않는다. 생성된 argv에 줄바꿈이나 NUL이 들어 있을 때만 막힌다(`source_validation.py:77-82`). 예외는 built-in handler가 따로 검사하는 경우다. 예를 들어 OpenPI는 dataset/normalization 경로가 절대경로인지만 확인한다(`adapters/__init__.py:1107-1109`).

주의:

- **Validate**는 repository가 정해지면 `resolve_runtime(inspection, {"backend": "auto"}, manifest.runtime)`을 돌린다(저장된 어댑터 `pipeline_api.py:3274-3286`, 저장 전 3323-3335). auto가 고르는 strong 후보는 세 가지뿐이다. 유효한 `.skynet.json/.toml`의 `runtime`, `uv.lock`과 `pyproject.toml`이 함께 있을 때의 uv, `conda-lock.yml/.yaml`이다(`source_control.py:1698-1773`). `uv.lock`만, `pyproject.toml`만, `environment.yml`, `requirements*.txt`, `Dockerfile`은 weak다(1751-1798). strong 후보가 없거나 어댑터 `allowed_backends`에 걸러지면 반드시 실패한다. 예를 들어 위 틀처럼 `["existing", "conda"]`인데 repo에 `uv.lock`만 있는 경우다. 오류는 "runtime auto-detection found no single strong runnable candidate: … Select a runtime explicitly and provide its lock/environment/image."다(`source_control.py:1850-1879`). strong backend가 둘 이상이면 "runtime auto-detection is ambiguous" 오류다.
  - Validate 요청에는 backend나 profile 칸이 없다(`pipeline_api.py:1364-1378`). 그래서 Validate 결과는 `INVALID`로 남는다(저장된 어댑터는 `FAILED`로 기록된다).
  - 학습 제출은 막히지 않는다. 이 기록을 읽는 제출 코드는 없다(`record_adapter_validation`은 저장만 한다, `database.py:3409-3457`). 제출에 `profile_id`나 auto가 아닌 backend가 있으면 `resolve_runtime`을 부르지 않고 inspection을 `skipped`로 남긴다(`pipeline_api.py:2679-2785`). 브라우저는 늘 backend(`type`)를 명시하고, profile을 고르면 `profile_id`도 보낸다(`app.js:7162-7166`).
  - 코드 결함: **New**/**Edit** 상태의 Validate는 `/api/adapters/validate`에 `repository`, `revision`을 보낸다(`app.js:5684-5689`). 서버 모델은 `repository_url`, `source_revision`만 받고 나머지를 버린다(`pipeline_api.py:1373-1378`). 그래서 입력한 URL과 revision 대신 manifest의 `default_repository`와 `main`을 검사한다. `default_repository`가 없으면 검사 없이 `VALID`다(3319-3323). 또 이 route는 결과를 `{"report": …}`로 감싸는데(10713-10716), UI는 최상위 `status`/`errors`만 본다(`app.js:5698-5703`). 그래서 `INVALID`도 성공 색으로 칠한다. 결과 JSON의 `report.status`를 직접 읽는다. 입력한 URL과 revision으로 검사하려면 저장한 뒤 ADAPTER DETAIL 화면에서 Validate한다(`/api/adapters/{id}/validate`, `app.js:5667-5683`).
- built-in을 브라우저에서 **Edit**하면 그 slug에는 코드 쪽 갱신이 더 이상 seed되지 않는다(`database.py:3366-3368`).
  - 실험용 변형은 **Duplicate**를 쓴다. Duplicate는 고른 버전의 manifest를 slug까지 그대로 복사해 새 어댑터 v1을 만든다(`database.py:3243-3282`). slug로 묶인 hook은 그대로 동작한다.
- 편집 권한: installation owner는 built-in을 편집할 수 있다. 다른 email workspace는 Duplicate만 된다(`docs/email-workspaces.md:11`).
- 실험은 manifest와 그 sha를 spec에 박는다(`resolve_adapter_plan`, `adapters/__init__.py:3548-3569`). 나중에 registry를 고쳐도 기존 run은 바뀌지 않는다.

어댑터 API(prefix `/api`):

| Route | line |
|---|---|
| `GET /api/adapters?include_archived=` | 10667 |
| `POST /api/adapters` | 10696 |
| `POST /api/adapters/validate` | 10713 |
| `GET` / `PUT /api/adapters/{id}`(`expected_latest_version`) | 10721 / 10735 |
| `POST /api/adapters/{id}/clone` | 10754 |
| `DELETE /api/adapters/{id}`(보관) / `POST .../restore` | 10770 / 10778 |
| `POST /api/adapters/{id}/validate` | 10786 |

- README:322-324의 `PATCH`, `POST /api/adapters/{id}/archive`, `GET /api/adapters/{id}/validations`는 존재하지 않는다.

### 3-3. `generic` "Custom structured command" (브라우저만, 아무 repo)

- 입력 field는 둘이다(`adapters/__init__.py:2897-2909`).
  - `native.argv`: 라벨 **Training command**, `string_list`, 필수
  - `native.resume_argv`: 라벨 **Resume arguments**, `string_list`
- 둘 다 **4. Training settings** → **Adapter-declared inputs**(`index.html:1191`)의 textarea("One value per line")에 넣는다. argv 토큰 하나를 한 줄에 쓴다. JSON도 shell 문자열도 아니다(줄 단위로 나눈다, `app.js:3910-3913, 4072-4076`).
  ```
  python
  -m
  <module>
  --data
  /coc/flash7/<user>/datasets/<입력>
  --out
  {{SKYNET_RUN_DIR}}/artifacts
  ```
  - **Training command**를 비우면 "Training command is required."로 막힌다(`app.js:4190-4196`).
  - JSON 목록을 한 줄에 붙여 넣으면 그 줄 전체가 토큰 하나가 된다. DB에 실제 예가 있다: `egowam-dexverse-shared-proprio-smoke`의 첫 run은 `native.argv[0]`이 `["env","DEXVERSE_REAL_TRAINING=1",…]` 문자열 하나였고 `FAILED`다(실패 원인이 이것이라는 것은 추정).
- **Advanced model settings** → **Advanced native values**에는 그 밖의 키만 쓴다. 예: `config.checkpoint_globs=["<glob>"]`
  - 여기에 `argv=`나 `resume_argv=`를 쓰면 "Advanced native key “argv” conflicts with declared field “argv”."로 막힌다(`app.js:4102-4105, 4227-4237`). 화면 도움말도 "Do not repeat a typed field above."라고 적는다(`index.html:1234`).
  - `config.<path>=v`는 `native.config`에 들어가고, 그 밖의 키는 `native.overrides`로 간다(`pipeline_api.py:1429-1458`).
- 토큰: generic의 argv는 manifest 템플릿을 거치지 않고 글자 그대로 넘어간다(`adapters/__init__.py:491-497`). 실행 시 wrapper가 바꾸는 것은 `{{SKYNET_RUN_DIR}}`, `{{SKYNET_SOURCE_DIR}}`, `{{SKYNET_RESUME_CHECKPOINT}}` 셋뿐이다(`slurm.py:53-68`). `{{tokens.source_dir}}` 같은 manifest 토큰은 generic에서는 바뀌지 않는다(코드 판독).
- **Resume arguments**는 대체 명령이 아니다. 학습 attempt에 checkpoint가 있으면 wrapper가 이 값을 argv 뒤에 덧붙인다(`slurm.py:944-946`). 예: `--resume` / `{{SKYNET_RESUME_CHECKPOINT}}` 두 줄.
- 자동 재개:
  - **6. Checkpoint and variants**의 "Auto-resume train/eval as a new attempt…"(`#checkpoint-auto-resume`)를 끄거나 **Resume arguments**를 채운다.
  - Auto-resume가 켜졌는데 비어 있으면 plan이 막힌다: "generic auto-resume requires native.resume_argv"(`adapters/__init__.py:493`)
- 그 밖의 동작:
  - GPU는 1~16개, backend는 모두 된다. `SKYNET_ASSIGNED_GPU_COUNT`가 설정된다.
  - data binding, progress, evaluation contract가 없다. 화면에는 "This adapter does not require a registered dataset."가 뜬다.
- 올바른 형태:
  - 코드는 cwd(고정 commit checkout의 project dir) 기준 상대 경로나 `{{SKYNET_SOURCE_DIR}}`로 가리킨다.
  - 절대 경로는 데이터와 출력에만 쓴다.
  - 환경을 명시한다: **Named runtime profile**, 또는 **Existing environment (manual)** + venv 경로.
- 선례 `lightwheel-*`는 따라 하지 않는다. source 고정을 우회했다.
  - experiment `lightwheel-yam-synthetic-v3-20260929`: `source.repository=https://github.com/Boey-li/EgoSim-Private`, `revision=384edad0…`, `dirty_policy=reject`, `runtime.backend=existing`, `environment_path` 없음, gateway sky2, partition rl2-lab, L40S 1개, `cpus_per_task 8`, 64 GB, `00:30:00`
  - generic run 18개 중 17개가 argv를 `/usr/bin/env --chdir=/coc/flash7/ycho420/repos/EgoSim-Private …`로 시작한다. Lightwheel 16개는 `environment_path`도 없어 node의 시스템 `python3`로 돌았다(`slurm.py:1366-1374`).
  - wrapper는 고정 cache(`/coc/flash7/ycho420/repos/EgoSim-Private-930e4a68/384edad0…`)에서 argv를 시작하는데(`slurm.py:961`), `--chdir`이 손으로 관리하는 checkout으로 옮겨 간다. sky1에서 그 checkout의 HEAD는 `96676ab96a3db52a94edfb724a6d7481059aaa29`이고 `git status --porcelain` 항목이 24개다. 즉 실제로 돈 코드는 기록된 commit `384edad0…`이 아니다.
  - 결과도 18개 중 FAILED 13, CANCELLED 3, SUCCEEDED 2다.

### 3-4. 새 built-in 어댑터 (코드 변경과 재시작)

필요한 경우: Skynet recording을 Convert해서 쓰거나, data bridge가 필요하거나, `dexverse_recorded` 평가를 받으려는 경우.

추가할 파일(HAT/DP를 본뜬다. `dp_manifest.py`는 72줄로 가장 짧은 예다):

| 파일 | 내용 | 본보기 |
|---|---|---|
| `skynet_app/adapters/<x>_manifest.py` | 상수 `REPOSITORY`, `REVISION`, `CONTRACT`, `RUNTIME`(`hat_manifest.py:10-13`), `support_files()`(capsule `adapter-support/*`, 16-29), `recording_conversion()`(conversion preset, `training_setup`, `loader_validation`, 31-45), `manifest()`(48-132) | `hat_manifest.py`, `dp_manifest.py` |
| `<x>_runtime.py` | 깨끗한 고정 revision만 허용(`hat_runtime.py:45-50`), upstream 경로를 `sys.path`에 추가(62), 공식 model/loss 호출, checkpoint·점수·`logs.json.txt` 기록(226-279) | `hat_runtime.py`, `dp_runtime.py` |
| `<x>_data.py` | recording dataset reader. `require_validation=True`(`hat_data.py:165-166`) | `hat_data.py`, `dp_data.py` |
| `<x>_evaluation.py` | recorded 시뮬레이터 평가 loader | `hat_evaluation.py`, `dp_evaluation.py` |
| 등록 | `builtin_adapter_manifests()`(`adapters/__init__.py:2885-2897`)에 추가 | |
| 평가 연결 | `RECORDED_POLICY_MODELS`(`evaluation_compatibility.py:16-24`), `adapters/policy_loading.py:4-19`, `compose_evaluator`(149-, evaluator profile은 `isaacsim-5.1.0_isaaclab-2.3.2_py311`로 고정, 186) | HAT 항목 |
| Convert 조건 | manifest `train.data_requirements.recording_conversion` 선언. `dataset_formats.py:12-43`은 보관되지 않은 모든 registry 어댑터(브라우저로 만든 것 포함)의 최신 manifest를 읽고 slug를 보지 않는다. 추가로 `loader_validation.runtime_profile`이 `skynet.json`에 있어야 하고(첫 prerequisite = upstream checkout, `policy_exports_cluster.py:52-68`), loader 코드는 `train.capsule_files`의 `adapter-support/` 아래에 있어야 한다(`policy_exports.py:385-391`) | HAT `recording_conversion()` |
| slug로 묶인 곳(HAT 전용) | `evaluation_targets.py:9-13`(평가 대상 데이터셋), `recording_sampling.py:42-46`(window policy, frame budget) | |
| runtime profile | `skynet.json` `runtime_profiles["<id>"]`. 코드에 박힌 id는 §2-3 끝 | §2-3 |
| 테스트 | `tests/test_hat_adapter.py`, `tests/test_dp_adapter.py`, `tests/test_hat_evaluation.py`, `tests/test_adapter_recording_contracts.py` | |

manifest `train`의 계약:

- input field(`AdapterInputField`, `adapters/__init__.py:1555-1648`):
  - `kind`는 `string | integer | number | boolean | json | string_list`다.
  - data_binding이 있는 field는 `default`를 가질 수 없다.
  - `value_path: "selection"`이면 `kind`는 `json`이어야 한다.
  - HAT의 데이터 입력은 `native.config.datasets`다: role `training_data`, `cardinality` many, format `skynet.recording-dataset/v1`, contracts `[skynet.hat-rgb-fingertips/v1]`, `value_path: "selection"`.
- 토큰(2645-2650):
  - `{{tokens.run_dir}}`, `{{tokens.source_dir}}`, `{{tokens.resume_checkpoint}}`, `{{computed.gpu_count}}`
  - spec 경로: `{{train.batch.value}}`, `{{native.config.x}}`, `{{identity.experiment}}`
  - 값이 없으면 blocker가 된다(2154-2163).
- `parameter_flags`의 `train.*` binding은 사용자가 그 값을 명시했을 때만 나간다(2177-2185).
  - 그래서 HAT는 `train.*` 값을 argv에 직접 쓴다.
- `strict_native_config: true`이면 모르는 `native.config` 키를 거부한다(`pipeline_api.py:2368-2374`).
- capsule files(265-305):
  - 상대 경로, 텍스트만, 파일당 1 MB 이하, 합계 4 MB 이하
  - 예약 이름: `argv.json`, `attempt-snapshot.json`, `checksums.sha256`, `execution.json`, `job.sbatch`, `native-config.json`, `preparation.json`, `requested-spec.json`, `resolved-spec.json`
  - 어댑터는 고정된 spec을 `{{tokens.run_dir}}/resolved-spec.json`에서 읽을 수 있다. HAT는 이것을 `--data-spec`으로 넘긴다.
- progress(159-233):
  - `unit`은 `step | epoch`, `total_path`는 `train.max_steps | native.config.epochs`다.
  - source는 둘 중 하나다.
    - `log_regex`: named group `completed`, `total`
    - `jsonl`: `path`(run dir 기준), `completed_key`, `required_key`, `metrics`
  - 자세한 내용은 `docs/training-metrics.md`에 있다.
- checkpoint:
  - glob은 `run_dir`과 `project_dir` 양쪽에서 찾는다(`slurm.py:447-470`).
  - `final_selector: "best"`이면 후보마다 점수 sidecar(schema `skynet.checkpoint-score/v1`)가 필요하다(`slurm.py:503-580`). 디렉터리 후보는 `<dir>/skynet-checkpoint.json`, 파일 후보는 `<file>.skynet-checkpoint.json`이다(`slurm.py:505-508`). HAT는 `hat_runtime.py:39-42`에서 쓴다.
- argv에 "submitit"이 들어가면 거부한다(`adapters/__init__.py:369-370`).
- defaults 우선순위: 명시 입력 > manifest defaults > cluster defaults(README:222, `pipeline_api.py:2595-2646`)

등록(seeding):

- 앱 시작 시 실행된다: `pipeline_api.py:1690` → `_seed_registries_unlocked`(1719-1738) → `upsert_seed_adapter`(`database.py:3319-3392`).
- manifest sha가 바뀌었고 최신 버전의 `created_by == "__seed__"`이면 새 버전을 붙인다.
- `database.workspace_id`가 설정되어 있으면 seeding을 건너뛴다.
- 그 slug를 브라우저에서 편집한 적이 있으면 seed되지 않는다(§3-2).

---

## 4. 데이터

### 4-1. 핵심 제약

- HAT, DP, ACT, EgoVerse ACT/HPT, XPolicyLab ACT Native는 `location.path` 또는 `selection`으로 데이터를 받는다.
  - 서버는 `data_locations` row(kind `cluster`, status `AVAILABLE`, manifest SHA 일치)가 없으면 거부한다: "Choose a verified training-cluster copy of every dataset"(`pipeline_api.py:2318`).
  - 이 row를 쓰는 코드는 Convert 게시 경로 하나뿐이다(`policy_exports_cluster.py:210` → `database.record_data_location`, `database.py:3819`). location을 만드는 HTTP endpoint는 없다.
- HF import와 **Add files**로 만든 버전은 `READY`만 되고 location, contract, validation이 없다.
  - 그래서 `version.path` + contracts 없는 binding만 받는다. 지금 그런 binding은 GR00T(`adapters/__init__.py:3137`)와 OpenPI(`adapters/__init__.py:3296`)뿐이다.
- 학습 데이터 후보는 category `dataset`이고 `data_dataset_presentations` row가 있는 버전뿐이다(`dataset_catalog.py:42-55`, `data_selection.py:105-108`). 이 row는 trigger가 category `dataset`일 때만 만든다(`skynet_app/migrations/postgresql/016_flat_dataset_catalog.sql:23-38`).
  - Data → **Files** 탭의 **New**는 category `file`(Simulation assets, Hand / robot assets, Calibration, Checkpoint, Model, Other files)로 등록한다(`app.js:18613-18626`, `data_resource_policy.py:4-19`). 그 버전은 **2. Training datasets**에 나오지 않고, 서버도 "Choose a dataset for training data; files belong to other inputs"로 거부한다(`data_selection.py:66-67`).
  - 학습용 HF import와 Add files는 반드시 Data → **Datasets** 탭에서 시작한다(§4-3, §4-4).
- 외부 native 데이터(EgoVerse zarr, XPolicyLab native, 노트의 LeRobot/HDF5)를 위 어댑터가 학습할 수 있게 하는 정식 경로는 없다.
  - `docs/egoverse-models.md:57-79`와 `docs/xpolicylab-adapters.md:36-59`는 manifest SHA, metadata `contract`/`validation`, 검증된 location을 등록하라고 한다.
  - 하지만 location을 기록하는 API가 없다. DB row를 손으로 쓰거나 개인 스크립트를 만들면 AGENTS.md 위반이다.

| 데이터 출처 | 정식 경로 | 쓸 수 있는 어댑터 |
|---|---|---|
| DexVerse 시뮬레이터 시연(직접 수집 또는 이미 import된 release session) | Data → Collect → Recordings → **Convert** | HAT, DP, ACT, EgoVerse, XPolicyLab ACT Native(어댑터가 `recording_conversion`을 선언한 경우) |
| 공개 Hugging Face 데이터셋 | Datasets → **New**(Provider `huggingface`) → **Register** → 자동으로 열리는 **Import from Hugging Face** | GR00T, OpenPI(format과 스키마가 맞을 때) |
| 이미 클러스터에 있는 파일 | Datasets → **New**(Provider `local`/`s3`/`http`) → **Register** → 자동으로 열리는 **Add files** | GR00T, OpenPI(format과 스키마가 맞을 때). 검증 없음 |
| 노트의 외부 데이터(`/coc/flash7/czhang883/...`, `/coc/flash7/zhenyang/...`) | 없음 | `generic` argv에 경로를 직접 넣는다(등록·검증 없음). 아니면 정식 등록·검증 기능을 먼저 코드로 만든다 |

### 4-2. Recordings → Convert

1. Data 탭(`index.html:2537-2577`: **Collect** | **Recordings** | **Datasets** | **Files**)에서 **Collect**로 수집한다(`docs/live-dexverse.md`).
   - 수집은 workstation `rl2-bonjour`에서 GPU teleop session으로 한다.
   - 에피소드마다 action T개와 scene state T+1개를 저장한다. 관측과 영상은 렌더하지 않는다.
   - archive는 크기와 SHA를 확인한 뒤 클러스터 `raw/dexverse-live/...`로 옮겨진다.
   - 이미 import된 release session도 있다: `floating_shadow_right` 24 session(1,200 recording), `floating_shadow_bimanual` 14 session(700 recording).
2. **Recordings**에서 session 행을 찾는다. 열은 Session, Recordings, Created at, Dataset, Actions이고(`index.html:2622-2630`) archive 상태 열은 없다. 행 버튼은 **View**, **Convert**, **Delete**다. 미리 확인할 것은 없다. archive가 아직 클러스터에 없으면 Convert job이 스스로 기다린다(4단계의 "Waiting for the recording archive").
3. **Convert**를 누르면 "Convert to a dataset" 창이 열린다(`index.html:4900-4975`).
   - **Dataset name**: 인쇄 가능한 문자 1~100자
   - **Adapter**: `"<name> · v<N>"`. `train.data_requirements.recording_conversion`이 없는 어댑터는 비활성이다.
   - **Input modalities**(읽기 전용)
   - **Validation episodes (%)**: 0~50, 기본 20.
     - 브라우저는 0%를 막지 않는다. 경고만 띄운다: "Warning: no validation recordings. Training will run without validation steps."(`policy-exports.js:106-127`). recording preset 중 `validation_required`를 켠 것이 없기 때문이다(기본 False, `training_contracts.py:39`. 검사는 `policy_exports.py:471`).
     - 하지만 HAT, DP, ACT Native의 학습·평가 loader는 `require_validation=True`로 sampling한다(`hat_data.py:166`, `dp_data.py:109`, `act_native_data.py:93,117`, `dp_manifest.py:66`). validation window가 비면 `recording_time.py:66`이 오류를 낸다. 따라서 이 어댑터들의 데이터는 반드시 0%보다 크게 Convert한다.
   - **Split seed**: 기본 42
   - 브라우저는 항상 어댑터의 **default** data preset을 보낸다(`policy-exports.js:58-61,572`).
4. **Convert**를 누른 뒤 Datasets → **Conversion history**에서 단계를 본다(`policy-exports.js:144-155`): "Queued" → "Preparing submission" → "Submitting CPU job" → "Waiting for the recording archive"(archive가 클러스터에 닿을 때까지) → "Checking originals" → "Preparing required observations" → "Converting" → "Validating" → "Checking adapter data loader" → "Prepared".
   - CPU job이다: queue `normal`, 4 CPU, 32 GB, 1 h(`policy_exports_cluster.py:72-117`). GPU가 없으므로 8 CPU/GPU 정책과 무관하다.
   - 게시하면 version(`ON_CLUSTER`), derivation, cluster location(`AVAILABLE`)이 기록된다.

서버 제약(`policy_exports.py:398-470`):

- 대상은 클러스터만이다.
- 브라우저와 API에서 Convert 한 번은 정확히 recording session 하나의 모든 에피소드다. `ExportRequest`는 `session_id` 하나만 받고 `extra="forbid"`다(`policy_exports_api.py:17-28`). 브라우저도 `session_id`만 보낸다(`policy-exports.js:563-577`).
  - 서비스 내부 제한은 서로 다른 session 1~25개다(`policy_exports.py:405-445`). 하지만 `selections`를 넘기는 호출자가 없어 브라우저로는 닿지 않는다.
  - 여러 session으로 학습하려면 session마다 Convert하고 **2. Training datasets**에서 여러 데이터셋을 고른다. binding이 `cardinality: many`인 어댑터(예: HAT, Cmd/Ctrl 다중 선택)만 된다.
- 에피소드 1,000개 이하, 원본 이미지 20 GB 이하, recording SHA 중복 없음
- 에피소드 수 ≥ preset `minimum_episodes`
- session의 robot이 preset `supported_robots`에 있어야 한다.
- split은 에피소드 SHA를 seed로 정렬해 정해지는 결정적 방식이다(311-337).
- 같은 설정으로 다시 누르면 기존 job을 재사용한다(fingerprint, 488-497).

HAT가 받는 손(`ops.datasets.action_codecs.unidex.supported_robots()`):

- `floating_shadow_right`, `skynet_allegro_v4_right`, `skynet_inspire_rh56_right`, `skynet_leap_v1_right`, `skynet_sharpa_right`, `skynet_wuji_1_right`, `skynet_wuji_2_right`
- 오른손만 받는다. `skynet_shadow_right`(직접 수집한 Shadow)와 `floating_shadow_bimanual`은 목록에 없다.
- WUJI1과 Sharpa mapping은 Skynet 확장이다(`docs/policy-data-exports.md:17`).

API: `POST /api/data/exports` `{session_id, adapter_id, adapter_version_id, adapter_data_preset, name, gateway, validation_percent, seed, overfit_episode?}`(`policy_exports_api.py:17-28,55`), `GET /api/data/exports/jobs`, `GET /api/data/exports/{id}`, `POST /api/data/exports/{id}/retry`

### 4-3. Hugging Face import

1. Data → **Datasets** 탭 → **New**를 누른다. "Register dataset or files" 창의 제목이 "New dataset"으로 바뀐다(`app.js:18613-18626`).
   - **Name**: 비워도 된다("Leave blank to use the source name.")
   - **Description**
   - **Source details** 펼침 안: **Provider** = `huggingface`, **Namespace** = HF org, **Source key** = repo 이름, **Type** = Demonstrations(기본) / Dataset / Evaluation data / Raw capture 중 하나(`data_resource_policy.py:5-10`)
   - **Register**(`index.html:4108-4114`)를 누른다.
2. **Import from Hugging Face** 창이 저절로 열린다(`app.js:15190-15198`, 창 `index.html:4217-4350`). 아래를 넣는다.
   - **Exact Hugging Face revision**: 40자 hex. "Mutable names such as main are rejected."
   - **Dataset directory**: repo 안의 디렉터리. glob과 `..`은 안 된다. repo 전체는 import할 수 없다.
   - **Format**: "Must exactly match the format declared by the consuming adapter."
   - **Bundle role**(필수): 학습 입력이면 `training_data`. 제안값은 `training_data`, `evaluation_data`, `simulation_assets`다(`index.html:4277-4283, 4893-4897`. `role`로 전송, `app.js:14807`).
   - **SSH gateway**, **Slurm queue**(`skynet.json` `queues`의 각 항목. 기본은 `defaults.import_queue_policy`, 지금은 **Preemptible / overcap**), **CPUs**(기본 8), **Memory / GB**(기본 32), **Time limit**(기본 `04:00:00`, "Slurm duration, from 1 minute to 24 hours.")
   - **Submit import job**을 누른다.
3. Data → Datasets(또는 Files) 패널 머리의 **Import history**를 연다. 이 버튼은 import job이 하나라도 생긴 뒤에야 보인다(`app.js:14691`).
   - 열: Dataset directory, Revision, State, Slurm, Dataset, Updated, Actions. 행 동작은 **Detail**(로그)과 **Cancel import**다(`app.js:14692-14706`).
   - 성공 상태는 `SUCCEEDED`다(DB `data_imports.state`).
   - CPU job이다. sbatch에 `--gres`가 없고 `--cpus-per-task`(**CPUs**)와 `--mem`(**Memory / GB**)만 잡는다(`data_imports.py:349-360`). **CPUs**는 병렬 다운로드 thread 수로도 쓰인다(`data_imports.py:127`).
   - Skynet은 Hugging Face token을 넘기지 않는다. Import 요청에 token 칸이 없고(`pipeline_api.py:1582-1593`), 제출할 때 환경변수도 전달하지 않는다(`pipeline_api.py:6689`). `huggingface_hub`는 `HF_TOKEN` 환경변수나 `$HF_HOME/token` 파일을 저절로 읽는다. 그런데 job은 `HF_HOME`을 cluster 설정 `paths.huggingface_cache`(`/coc/flash7/ycho420/.cache/huggingface`)로 바꾸고(`data_imports.py:367`), 그 자리에는 token 파일이 없다. job 스크립트에 `--export`가 없어 sbatch를 부른 login 환경을 물려받지만, sky1/sky2에서 `HF_TOKEN`은 비어 있다. `~/.cache/huggingface/token`은 있지만 이 경로 밖이라 읽히지 않는다(2026-10-06 확인). 그래서 지금은 익명으로 받는다. 공개이면서 gated가 아닌 dataset repo만 import되고, private·gated repo는 job이 실패한다.
   - 버전 revision은 `<rev>#subset=<dir>`, 상태는 `READY`다.
4. 같은 source에서 나중에 다시 import하려면 Datasets → **New**를 같은 Provider/Namespace/Source key/Type으로 다시 한다. "Using the existing source registration." toast가 뜨고 Import 창이 다시 열린다(`app.js:15177-15198`). Datasets 행에는 **Import** 버튼이 없다.

### 4-4. Add files (이미 클러스터에 있는 파일)

- 학습 데이터면 Data → **Datasets** → **New**에서 **Provider**를 `local`(또는 `s3`/`http`)로 두고 **Register**를 누른다. **Add files** 창이 저절로 열린다(`app.js:15190-15198`).
  - 입력: **Upstream revision**, **Format**, **Cluster path**, **Source URI**, **Manifest SHA-256**(64자 hex, 필수), **Status**(Ready / Staging / Unavailable), **Size / bytes**
  - **Create immutable version**(`index.html:4211`)을 누른다.
- **Import**와 **Add files** 행 버튼은 Files 탭 행에만 있다(`app.js:14640-14642`). Files 탭에서 만든 버전은 category `file`이라 학습 데이터로 쓸 수 없다(§4-1). Datasets 쪽 source에 버전을 더하려면 같은 source로 **New**를 다시 한다(§4-3 4단계).
- 경로, manifest, SHA가 실제로 있는지는 아무도 검사하지 않는다(추정, 코드 판독).
- README가 요구하는 레이아웃(`README.md:411-447`): `manifest.json`, `inventory.sha256`, `data/`, manifest SHA를 담은 `READY` 파일

### 4-5. 실험에서 데이터 고르기

- **2. Training datasets**(`index.html:992`)에서 고른다.
  - HAT처럼 binding이 `cardinality: many`이면 다중 선택이 된다. Cmd/Ctrl로 고르며 고른 순서를 유지한다.
  - 후보는 `GET /api/data/selections`에서 온다: category `dataset`이고, 보관(archived)·retired가 아니며, 검증된 location이 있거나 `READY`인 버전(`dataset_catalog.py:42-55`, `data_selection.py:105-108`)
  - 맞지 않는 데이터셋은 비활성이고 이유가 표시된다.
- Datasets 탭의 **Use in experiment**를 쓰면 Convert에 쓴 어댑터 버전 그대로 Submit이 열린다(`app.js:20957-21030`).
- 서버 추가 검사(`adapters/dataset_inputs.py:15-46`): 데이터셋 중복 없음, 데이터셋 사이 원본 recording 중복 없음, train/validation 누수 없음
- 주파수:
  - **Control frequency (Hz)**를 비우면 원본 주파수를 쓴다.
  - 정수 downsampling만 된다. 값은 원본 주파수를 정확히 나눠야 한다(`adapters/recording_time.py:21-31,73-105`). 예: 60 Hz 원본이면 60, 30, 20 등.
  - 오류 문장: "…must divide the recording frequency … exactly; interpolation is not supported"
- **Action chunk**가 window를 정한다.

### 4-6. 평가 대상 데이터 (HAT만)

- Evaluations → Submit의 **Evaluation dataset**과 **Unseen hand**는 suite가 `requires_target_dataset`일 때만 나온다.
  - 지금은 `human-policy-hat` → `skynet.hat-rgb-fingertips/v1`만 해당한다(`evaluation_targets.py:9-13`).
- 대상 조건: HAT contract, format `skynet.recording-dataset/v1`, `PASSED`, 손 정확히 하나
- 지금 쓸 수 있는 HAT 데이터셋 7개는 모두 `ON_CLUSTER`, `AVAILABLE`, 51 episodes, train 41 / validation 10(seed 42, 20%), 60 Hz다.

| Version id | Display name |
|---|---|
| `3ef56db5-c8f0-4b7d-b30a-d19cbebac5cb` | skynet_inspire_rh56_right · PickCube · HAT · 51 episodes |
| `cf75c50c…` | skynet_allegro_v4_right · PickCube · HAT · 51 episodes |
| `b78f3644…` | skynet_wuji_1_right · PickCube · HAT · 51 episodes |
| `bd25c3e0…` | skynet_sharpa_right · PickCube · HAT · 51 episodes |
| `7327b23d…` | skynet_leap_v1_right · PickCube · HAT · 51 episodes |
| `f3b6d68c…` | floating_shadow_right · PickCube · HAT · 51 episodes |
| `71599257…` | WUJI2 · Pick up cube · HAT · 51 episodes |

### 4-7. 하지 않는 것

- `ops/datasets/import_dexverse_release.py`를 손으로 돌리지 않는다. 앱 진입점이 없는 클러스터 worker이고, 손으로 돌리면 개인 실행 경로가 된다.
- `data_resources`, `data_resource_versions`, `data_locations`에 row를 손으로 넣지 않는다.

---

## 5. 실험 정의와 제출

### 5-1. Submit 폼

위치는 Experiments → **Submit**("Configure a training run", `index.html:886`)이다. 앞 단계가 유효해야 다음 fieldset이 열린다(`docs/training-contracts.md:71-77`).

| 단계 | 주요 입력 |
|---|---|
| **1. Algorithm and code** | **Experiment name**(`[A-Za-z0-9][A-Za-z0-9_.-]*`, 96자 이하), **Algorithm / adapter**, **Repository**와 **Refresh refs**, **Branch**, **Commit**("The exact selected SHA will be recorded."), **Project subdirectory**(기본 `.`). **Commit**은 입력 칸이 아니라 `<select>`다(아래) |
| **2. Training datasets** | §4-5 |
| **3. Runtime** | **Runtime**, **Named runtime profile**, custom 경로, **Runtime details** |
| **4. Training settings** | **Learning rate**, **Batch size**, **Batch semantics**, **Gradient accumulation**, **Data workers**, **Precision**, **Max steps / iterations**, "Adapter-declared inputs", "Training preset", **Inputs and outputs**, **Advanced model settings → Advanced native values** |
| **5. Resources** | **Queue policy**(**Auto**와 `skynet.json` `queues`의 각 항목. 지금은 **Normal / rl2-lab**, **Preemptible / overcap**), **GPU allocation**(Auto / Manual), **GPUs / node**(Manual일 때만), **GPU type**(`gpu_aliases`의 각 항목. 지금은 Any compatible / A40 / L40S / RTX 6000), **Nodes**(1 고정), **CPUs / task**(읽기 전용), **Memory / GB**(기본 64), **Wall time**(기본 `04:00:00`) |
| **6. Checkpoint and variants** | **Checkpoint**(Start fresh / Resume full state / Initialize weights), **Checkpoint path or run ID**, **Save every optimizer steps**, **Checkpoint warning / seconds before timeout**(기본 300), **Maximum automatic attempts**(기본 5), Auto-resume 체크박스, **Sweep / variant matrix** |
| **7. Tracking and evaluation** | **Weights & Biases**(기본 켜짐), **MLflow**, **W&B project**(기본은 실험 이름), **W&B run name template**(기본 `{experiment}/{variant}/run-{run_number}`), "Record downstream evaluation suites in experiment metadata" |

- **Commit** 고르기(`index.html:957-967`, `app.js:6865`):
  - 목록은 `GET /api/source/commits?repo_url=…&branch=<Branch>&limit=50`에서 온다(서버 최대 100, `pipeline_api.py:10816-10821`). 즉 **Refresh refs** 뒤 **Branch**에서 고른 원격 branch의 최신 50개 commit만 고를 수 있다. manifest의 기본 revision으로 채우는 경로도 없다(`app.js:2337-2366`).
  - 목록 밖의 commit은 `installPinnedSourceRevision`(`app.js:8680-8712`)으로만 고정된다. 이 함수는 Presets → **Load configuration**(`app.js:8977`)과 Datasets → **Use in experiment**(Convert한 데이터셋, `app.js:21077`)에서만 불린다.
  - 따라서 fork commit은 원격 branch에 push하고, 그 branch의 최신 50개 안에 있을 때 고른다.
- 7단계의 evaluation suite는 metadata일 뿐이다. 화면 문구: "Runnable jobs are created in the Evaluations tab"(`index.html:1548`)
- **Memory / GB** 기본은 브라우저 64이고 cluster `defaults.memory_gb`는 16이다. 필요한 값을 직접 넣는다.

### 5-2. "preset"은 두 가지다

| 이름 | 무엇 | 어디 |
|---|---|---|
| Experiment preset | 저장된 Skynet 실험(`DRAFT` 또는 `SUBMITTED`) | Experiments → **Presets** 탭. **New**는 "New experiment preset" 창을 열고 버튼은 **Create preset**이다(제출 안 함). 행 동작은 **Load configuration**, **Submit**(DRAFT) 또는 **View variants**(SUBMITTED), **Delete** |
| 어댑터 `train.presets` | manifest 안의 버전 붙은 기본값 묶음 | **4. Training settings**의 "Training preset". 값을 하나라도 바꾸면 `custom`이 된다(`app.js:3731-3784`) |

HAT의 `train.presets`(`adapters/hat_manifest.py:77-95`):

| preset | 내용 |
|---|---|
| `hat-fit/v1` | 3000 updates, batch 16, LR 1e-4, chunk 50, seed 42 |
| `hat-fit-resnet/v1` | |
| `hat-seven-hands-main/v1` | 8000 updates, 1800 source frames, 30 Hz / chunk 30, `hand_balanced`, final selector `latest`, validation off |

### 5-3. sweep (반복 seed, 설정 행렬)

- 변환 규칙(`pipeline_api.py:1461-1489`):
  - `"seed"` 또는 `"seeds": [...]`는 `sweep.seeds`가 된다.
  - `train.`, `resources.`, `runtime.`, `tracking.`, `native.`, `data.`로 시작하는 키는 경로 그대로 쓴다.
  - 그 밖의 키는 `native.overrides.<key>`가 된다.
- 고정값: strategy `grid`, `max_parallel` 2. variant가 20개를 넘으면 preview에 경고가 뜬다.
- variant 이름은 `v{index:03d}-s{seed}-{sha8}`이다(`experiments.py:868-905`).
- `sweep.seeds`와 `train.seed` 축은 함께 쓸 수 없다(601-603).
- repo에서 찾은 choice field는 sweep할 수 없다(`pipeline_api.py:2553-2580`).
- 권장(`docs/training-contracts.md:100-117`):
  - seed 반복은 preset과 sweep으로 한다.
  - 손 subset은 subset마다 별도 preset이나 실험으로 만들어 학습 구성을 검토할 수 있게 한다.
  - 1~6손 비교에서는 평가 대상을 고정한다.
  - HAT 본 실험은 (subset, seed)마다 실험 하나였다: `hat-main-k<k>-<subset-hash>-s1701..s1703`

### 5-4. 자원과 8 CPU/GPU

- 설정 위치: `skynet.json:536` `defaults.cpus_per_gpu: 8`, `cpus_for_gpus()`(`cluster_config.py:407-411`)
- 서버가 학습·평가 sbatch를 만들 때 강제한다(`slurm.py:1442`, `pipeline_api.py:4811-4818`).
  - 바뀌면 `execution_provenance`에 `cluster_cpu_per_gpu_policy` 변환으로 남는다.
  - 예전 spec에도 적용된다. HAT 본 실험 spec은 `cpus_per_task: 2`로 저장되어 있다.
- UI의 **CPUs / task**는 읽기 전용이고 `const CPUS_PER_GPU = 8`(`app.js:2562-2575`)로 계산한다. 이 값은 config를 따르지 않는 중복 상수다.
- queue는 `skynet.json` `queues`가 정한다. 코드와 화면은 이 표를 복사하지 않고 설정을 읽는다(가드 테스트 `tests/test_no_hardcoded_cluster_facts.py`). 2026-10-06 기준 값:

  | queue | account/partition | 최대 시간 | 선점 |
  |---|---|---|---|
  | `normal` | `rl2-lab`/`rl2-lab` | 4 h | 아님 |
  | `overcap` | `overcap`/`overcap` | 2일 | 됨 |

- **Queue policy** Auto(`_auto_queue`, `pipeline_api.py:4277`):
  - wall time이 normal 한도를 넘으면 overcap을 쓴다.
  - 아니면 `gpu_usage -l`로 `rl2-lab` 할당량을 읽고, 넘치면 overcap을 쓴다.
  - 확인할 수 없으면 실패한다: "auto queue selection could not verify live GPU quota"
- GPU alias는 `any`, `a40`, `l40s`, `rtx_6000`이다(`skynet.json` `gpu_aliases`). 새 alias나 queue는 `skynet.json`에만 추가하면 된다.
  - 서버는 config를 따른다. gpu_type·queue_policy 검증(`experiments.py:449-454, 529-534`)과 queue별 account/partition 결정(`pipeline_api.py:3488-3490, 3641-3643`)이 `CLUSTER.gpu_aliases`와 `CLUSTER.queues`를 읽는다.
  - 폼의 **Queue policy**, **GPU type**, 데이터 import의 **Slurm queue**, 수집 양식의 account/partition/GPU 기본값, 대시보드 계정 사용량 열은 모두 서버가 페이지를 내보낼 때 config로 채운다(`skynet_app/page_markup.py`, `main.py` `index()`). gateway 선택지와 같은 방식이다. 평가 폼은 학습 폼의 GPU 옵션을 복제하고 suite의 `allowed_gpu_types`로 거른다(`app.js` `updateEvaluationGpuOptions`). Isaac suite에서는 그 alias가 `isaac_evaluation_placement` node의 `gpu_type`이어야 보인다.

### 5-5. 제출

액션 바 버튼(`index.html:1553-1580`): **Preview sbatch**, **Create draft**(저장만, 제출 안 함), **Create and submit**. preset 모드에서는 **Create draft**가 **Create preset**이 되고 **Create and submit**은 숨는다(`app.js:7680-7692`).

1. **Preview sbatch**를 누른다. 오른쪽 "Sbatch preview" 패널에서 warnings, **Script variant**, **Blocked variants**를 본다. **Download exact script**는 그 패널의 "Script options" 펼침 안에 있다(`index.html:1612-1622`).
2. **Create and submit**을 누른다(저장만 하려면 **Create draft**).
   - preview가 지금 폼과 다르면 거부된다: "Preview the current configuration before submitting."
   - 확인 창이 뜬다: "...The submitted specification will be locked."
   - 같은 이름이 있으면 버튼이 **Create draft revision** / **Create revision and submit**으로 바뀐다. 같은 spec이면 "this exact specification already exists as experiment revision N"으로 거부된다.
3. 제출되면 Experiments → **Runs**에 그 run이 열린다.

브라우저가 부르는 API 순서(`app.js:7762, 8384-8426`): `POST /api/experiments/preview` → `POST /api/experiments`(같은 이름이면 `POST /api/experiments/{id}/revisions`, `pipeline_api.py:11142`) → `POST /api/experiments/{id}/submit` `{gateway}`. **Create draft**는 마지막 submit을 부르지 않는다.

서버에서 일어나는 일(`_submit_stage`, `pipeline_api.py:4627`):

- 데이터 확인 → 어댑터 plan → CPU 정책 → 인자 검증 → queue 선택 → `compile_sbatch`
- capsule을 run 디렉터리 아래 `submissions/capsules/<sha256>`에 올린다. 새 run의 디렉터리는 `{paths.jobs}/runs/<run-id>`이고 `{paths.jobs}`는 workspace base path 기준이다(§1-3).
- 같은 gateway에서 `sbatch --test-only`, 그다음 `sbatch --parsable`을 한 번 실행한다.
- blocker가 있으면 `BLOCKED`(event `ADAPTER_BLOCKED`), 성공하면 `SUBMITTED`(event `JOB_SUBMITTED`)가 된다.
- 응답을 잃으면 `SUBMITTING`으로 남고 UI에 "SUBMISSION UNCONFIRMED"로 보인다.
- 동시 실행은 `max_parallel` 2다. 나머지 stage는 백그라운드 reconcile이 차례로 보낸다(`pipeline_api.py:4434`, 7592-7597).

sbatch directive(`slurm.py:1520-1544`):

- `--account`/`--partition`, `--nodes=1 --ntasks=1 --gres=gpu[:<alias>]:N --cpus-per-task=8*N --mem --time`
- `--chdir={paths.workspace}`
- 로그: `{paths.logs}/%x-%j.{out,err}`
- `paths.*`는 `compile_sbatch`가 workspace base path로 rebase한 값이다(`slurm.py:1406-1407`, `workspace_storage.py:17-48`). 지금 legacy root면 `/coc/flash7/ycho420/workspace`, `/coc/flash7/ycho420/logs`이고, HAT 본 실험 run은 `/coc/flash2/ycho420/skynet/...` 아래였다.
- `--export=NIL`
- 자동 재개가 켜진 학습에는 `--signal=B:USR1@<warning>`과 `--requeue`가 붙는다.
- 본문은 EXIT trap(final.json 기록) 설치 직후, capsule 검증이나 소스 준비보다 먼저 GPU preflight를 실행한다(`gpu_preflight.py`, 6장 참고).

job 환경변수: `SKYNET_RUN_ID`, `SKYNET_RUN_DIR`, `SKYNET_SOURCE_DIR`, `SKYNET_PROJECT_DIR`, `SKYNET_CHECKPOINT_DIR`, `SKYNET_CHECKPOINT_SAVE_STEPS`, `SKYNET_CHECKPOINT_KEEP_LAST`, `SKYNET_EVAL_PROGRESS_PATH`, `SKYNET_EVALUATION_RESUME`, `HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR`(`slurm.py:1545-1567`)

### 5-6. HAT는 브라우저로

- AGENTS.md 규칙이다. 서버는 이를 강제하지 않으므로 지키는 쪽의 책임이다.
- 브라우저에서 HAT를 고르면 아래처럼 동작한다(`hat_manifest.py:101-131`).
  - **Training datasets**가 다중 선택이 된다.
  - runtime profile은 자동으로 고르지 않는다. manifest에는 `runtime.allowed_backends={existing, conda}`, 권장 `existing`만 있다(`hat_manifest.py:105-106`). profile id `human-policy-hat`은 `recording_conversion()`의 `training_setup`과 `loader_validation`에만 있다(40-42).
    - Submit에서 시작하면 **3. Runtime**의 **Runtime**을 `Existing environment`(자동 감지가 없으면 `Existing environment (manual)`)로 두고, **Named runtime profile**에서 `Human Policy · HAT (shared ACT Python runtime) / configured`를 직접 고른다.
    - profile이 자동으로 채워지는 경로는 Datasets → **Use in experiment**(`app.js:21080-21086`)와 Presets → **Load configuration**(`app.js:9160-9174`)뿐이다.
  - Auto-resume가 자동으로 꺼진다(`supports_resume=False`, `app.js:4546`).
  - GPU는 1개만 되고 batch semantics는 `per_device`만 된다.
  - progress는 `artifacts/logs.json.txt`에서 읽는다.

### 5-7. 제출 전 확인

- [ ] Commit이 40자 sha이고 노트에 적은 원본 revision과 같다. 그 commit이 고른 Branch의 최신 50개 안에 있다(아니면 Load configuration으로만 재사용된다).
- [ ] **Named runtime profile**이 의도한 것이다. 평가까지 할 거면 평가 계획의 evaluator profile이 `runtime verified`다(§2-4).
- [ ] 데이터셋이 비활성이 아니다. Control frequency가 원본 주파수를 정확히 나눈다.
- [ ] Training preset 값이 노트의 설계와 같다. 바꿨으면 `custom`이 된 것을 노트에 적는다.
- [ ] **CPUs / task**가 GPU 수 × 8이다.
- [ ] Wall time이 queue 한도 안이다(normal 4 h).
- [ ] W&B가 연결되어 있고 project 이름이 노트와 같다.
- [ ] generic이면 **Adapter-declared inputs** → **Training command**에 한 줄에 토큰 하나씩 넣었고, Auto-resume를 껐거나 **Resume arguments**를 채웠다. argv에 손으로 관리하는 checkout으로 가는 `--chdir`이 없다.
- [ ] preview의 warnings와 Blocked variants가 비었다.

---

## 6. 모니터링, 평가, checkpoint

- GPU 미할당 시작(2026-10-06 확인): 컨트롤러가 `--gres`를 받고도 GPU 없이 job을 시작하는 일이 간헐적으로 있다(같은 날 다른 사용자 잡 103개 중 3개). 헤더나 GPU 종류와 무관한 클러스터 결함이므로 제출 양식을 바꿀 필요는 없다. 모든 GPU job 스크립트는 workload 전에 GPU preflight(`skynet_app/gpu_preflight.py`)를 실행해 `CUDA_VISIBLE_DEVICES`/`SLURM_JOB_GPUS`에 보이는 GPU가 요청 수보다 적으면 exit 97과 receipt(`gpu_not_allocated`)로 즉시 끝낸다. 학습·평가 attempt는 reconcile이 `BOOT_FAIL`(사유 "The cluster started the job without the requested GPU")로 분류해 `max_attempts` 안에서 자동 재제출하고, Runs의 attempt 표 Error 칸과 attempt 상세의 Scheduler reason에 그 사유가 보인다. 제출 양식에는 노드 제외 옵션이 없고 **Node**는 특정 노드를 고정할 때만 쓴다. accounting(`sacct`)이 꺼져 있는 동안은 job이 slurmctld 메모리에 있을 때 `scontrol show job <id>`로만 `AllocTRES`를 확인할 수 있다.

### 6-1. Runs

- Experiments → **Runs** "Training run history"(`index.html:1908`)
  - 열: Run, Experiment / variant, Training data, Run state, Slurm, Resources, Latest attempt, Progress, Started, Elapsed, Tracking
  - 행 동작: **View**, **Cancel**(실행 중), **Delete**
- **Run state** filter: All, Running, Pending, Submitting, Submitted, Cancelling, Retry pending, Submission failed, Blocked, Failed, Completed(DB 값 `SUCCEEDED`), Preempted, Draft, Cancelled, Timed out
- Training run 창(`index.html:5023-5066`):
  - **Checkpoints** 표: Checkpoint, Step, State, Created, Storage path
  - 동작(`app.js:11407-11414`): **Recover unconfirmed submission**, **Start evaluation**, **Connect W&B**, **Resume Training Run / new attempt**(checkpoint가 없거나 resume을 지원하지 않으면 **Restart training from beginning / new attempt**), **Start new Training Run from pinned variant**, **Cancel Training Run**
  - Attempts 표의 **Detail**에서 Common hyperparameters, Inputs and outputs, Adapter configuration, **stdout**, **stderr**를 본다.
- 성공한 run은 resume할 수 없고 rerun만 된다. clean retry는 deprecated다(`pipeline_api.py:4720`).
- API: `GET /api/runs/{id}/logs`, `GET /api/runs/{id}/attempts/{attempt_id}/logs`, `POST /api/runs/{id}/resume`, `/rerun`, `/cancel`, `/recover-submission`
- run 디렉터리: 그 run에 기록된 `runs.run_directory`다. Runs → **View**의 "Output storage path"(`app.js:11330`)에 보인다. 지금 `paths.work_root`에서 계산하지 않는다.
  - 안에는 `job.sbatch`, `attempts/<slurm-job-id>/`, `submissions/`, `resolved-spec.json`이 있다.
  - 새 run은 `{paths.jobs}/runs/<run-id>`(workspace base path 기준)다. `human-policy-hat` run 211개(hat-main 210개 전부 포함)는 `/coc/flash2/ycho420/skynet/jobs/runs/<run-id>`에 있다(DB 조회, sky1 확인).

### 6-2. progress, W&B, 알림

- progress와 W&B 전송은 백엔드 서버가 떠 있어야 된다. 브라우저는 닫아도 된다(`docs/training-metrics.md:47-49`).
- W&B에는 선언된 JSONL source의 유한한 scalar가 모두 간다. `training/{completed,total,...}` 내부 counter는 가지 않는다.
- Slack 알림은 Slurm job id가 확인된 attempt만 보낸다(`docs/slack-notifications.md`).

### 6-3. 자동 재개와 상태 불명

- 자동 재개 대상: `PREEMPTED`, `TIMEOUT`, `NODE_FAIL`, `BOOT_FAIL`, `REVOKED`(`preparation_states.py`의 `TRANSIENT_STATES`; reconcile과 invariant repair가 같이 쓴다)
  - 조건: auto_resume 켜짐, attempt 수 < `max_attempts`, 취소 요청 없음
  - 그러면 `RETRY_PENDING`이 된다.
- 시간 제한 종료는 exit `124:0`과 wrapper receipt가 맞을 때만 TIMEOUT으로 본다(`docs/time-limit-recovery.md`).
- 클러스터가 GPU 없이 잡을 시작하면 스크립트 맨 앞의 GPU preflight가 workload 전에 종료하고 receipt(`gpu_not_allocated`)를 남긴다(`skynet_app/gpu_preflight.py`). reconcile은 이를 `BOOT_FAIL`(사유: "The cluster started the job without the requested GPU")로 분류해 같은 조건으로 자동 재제출한다.
- Slurm 두 출처가 모두 job을 잊으면 `attempts/<job>/final.json`으로 정리한다.
  - exit 기록이 없으면 "Slurm no longer lists this job and it left no exit record. Its outcome is unknown: cancel it, or wait for Slurm accounting."로 멈추고 600 s마다 다시 본다.

### 6-4. checkpoint

- 학습이 성공하면 `checkpoint_globs`를 색인한다.
- 추론용 checkpoint는 `train.checkpoint.final_selector`(`best | latest`)로 고른다. 결과는 `checkpoints/selected-for-inference.json`이다(README:474-486).
  - 폼에는 이 값이 없다. 어댑터 기본값(HAT `best`)이나 preset(`hat-seven-hands-main` `latest`)에서 온다.
- **Start evaluation**은 `is_selected_for_inference`인 `AVAILABLE` checkpoint가 있을 때만 켜진다. 없으면 "A registered checkpoint is required"(`app.js:11998-12055`).

### 6-5. 평가 제출

- 시작점은 두 가지다.
  - Runs → **View** → **Start evaluation**: run id, checkpoint, suite가 채워진다.
  - Evaluations → **Submit**("Submit an evaluation", `index.html:2053`)
- **Evaluation target**: **Training run ID**, **Checkpoint path override**, HAT이면 **Evaluation dataset**과 **Unseen hand**(기본 체크), **Suite**, **Environment**(읽기 전용)
- **Execution controls**:
  - **Episodes / task**: 기본 20. 상한은 suite의 `maximum_episodes_per_task`(`app.js:12170-12172`)
  - **Tasks**: Select all / Use suite default
  - **Seeds**: 기본 `0,1,2`
  - **Parallel jobs**: 1부터 suite의 `maximum_parallelism`까지(`dexverse_recorded`는 8, `evaluation_compatibility.py:185`). 1이면 읽기 전용이고 도움말이 "This evaluator runs one worker."로 바뀐다. 1보다 크면 "One GPU per worker, in one Slurm allocation."(`app.js:12174-12182`)
  - **Maximum automatic attempts**, **Headless rendering**, **Auto-resume from episode ledger**
- **Slurm resources**: **Queue policy**(학습 폼과 같은 세 가지), **GPU type**(기본 `a40`, Any 없음. suite의 `allowed_gpu_types`로 걸러진다), **CPUs / worker**(읽기 전용 8), **Memory / worker (GB)**, **Wall time**(`app.js:12185-12230`)
  - Isaac suite의 `allowed_gpu_types`는 `isaac_evaluation_placement` node에서 온다: `l40s`(grom), `a40`(megazord)뿐이다(`pipeline_api.py:11054-11066`, `skynet.json:5-17`). 따라서 GPU type은 **A40**, **L40S**만 보인다.
- **Advanced → Manual evaluator command override**: **Evaluator argv / optional**, **Resume argv / optional**. 미검증으로 기록된다.
- **Submit evaluation**은 `POST /api/evaluations/validate-target`이 통과해야 켜진다.
  - `POST /api/evaluations`가 생성과 제출을 함께 한다. 별도 submit route는 없다.

제약:

- 새 평가가 막히는 것은 그 run에 활성 학습 stage가 있거나, 실행 파일을 공유하는 예전 평가가 돌 때뿐이다(`_evaluation_busy_reason`, `pipeline_api.py:900-909`). 새 평가는 항상 `execution_key=stage_id`로 계획되므로(`pipeline_api.py:10312`) 서로 막지 않는다.
  - 즉 run 하나의 여러 대상 손 평가를 동시에 제출하고 돌릴 수 있다. DB에도 실제 예가 있다: `hat-main-k1-bc20857d0948-s1701`의 평가 `163bf99c`(Slurm 3949785, 9/25 06:28:37→07:18:47)와 `f08beb74`(Slurm 3949800, 06:31:40→07:46:21)가 겹쳐 돌았다.
- Isaac 배치: L40S → `grom`, A40 → `megazord`(`skynet.json:5-17`). 설정의 `default_node`는 `grom`이지만 평가 폼에서는 `any`를 고를 수 없다. 바쁘면 기다리고 다른 노드로 넘어가지 않는다.
- episode ledger 크기 = tasks × seeds × episodes_per_task다. 완료된 `(checkpoint, suite, version, task, seed, episode_index)`는 재개 때 건너뛴다.
- Unseen hand(`evaluation_targets.py:34-103`):
  - run의 학습 입력에 있는 손은 거부한다: "The evaluation hand occurs in this run's training inputs; choose a held-out hand or disable Unseen hand"
  - 손 하나로 학습한 HAT run을 같은 손에서 평가하려면 **Evaluation dataset**에서 학습에 쓴 그 데이터셋을 고르고 **Unseen hand**를 끈다. Unseen hand가 꺼져 있으면 서버는 학습 입력과 손이 겹치는지 검사하지 않는다(`evaluation_targets.py:54-58, 91-103`). 켜 두면 위 거부 메시지가 나온다.
    - 목록에는 `GET /api/data/selections`의 데이터셋 가운데 assignment가 하나이고 contract가 `skynet.hat-rgb-fingertips/v1`인 것만 나온다(`app.js:5800-5816`, `data_selection.py:101-125`, `evaluation_targets.py:9-13`). 보관(archived)되지 않은 학습 데이터셋이면 여기에 나온다. 그 데이터셋에는 손이 정확히 하나 있어야 하고(`evaluation_targets.py:42-44`), 검증된 cluster 사본이 있어야 한다(`evaluation_targets.py:96-98`).
    - 서버에는 대상 없이 보내는 경로도 있다. `target_dataset_id`가 비어 있고 Unseen hand가 꺼져 있으며 학습 데이터셋이 정확히 하나면, 그 학습 데이터셋을 대상으로 쓴다(`evaluation_targets.py:79-86`). 학습 데이터셋이 둘 이상이거나 Unseen hand가 켜져 있으면 "Choose a separate HAT evaluation target hand"로 거부한다.
    - 하지만 브라우저 폼에서는 이 경로를 쓸 수 없다. HAT run이면 **Evaluation dataset**이 `required`가 된다(`app.js:5818-5825`, `requires_target_dataset`는 `pipeline_api.py:11069`). 폼이 유효하지 않으면 **Submit evaluation**은 꺼져 있다(`app.js:12331-12342, 13421`). 그러니 빈칸으로 두지 말고 같은 데이터셋을 고른다.
  - `diffusion-policy`는 학습한 손 하나에서만 평가하고, Unseen hand는 거부한다.

### 6-6. 결과

- Evaluations → **Runs** "Evaluation run history"(`index.html:2389`)
  - 열: Evaluation, Training run / checkpoint, Suite / environment, State, Execution, Progress, Task result, Started, Elapsed
  - 행 동작: **Results**, 그리고 활성 상태면 **Cancel**, 끝난 상태면 **Delete**(`app.js:13028-13047, 13073`)
  - 목록에는 대상 손이 나오지 않는다(열: 평가 이름 suite·짧은 id, run/checkpoint, suite/environment, 상태, 진행, 결과. `app.js:12971-13020`). 대상 손은 **Results** → **Episode results** → **Detail**의 episode viewer `robot` 값으로 본다(`GET /api/evaluations/{id}/episodes/{episode_id}/viewer`, `pipeline_api.py:12290-12292`).
- **Results** 창의 **Episode results** 표 → **Detail**에서 episode viewer, 영상, stdout/stderr를 본다.
- **Results** 창의 동작(`app.js:13840-13855`, 조건 `pipeline_api.py:1975-2017`):
  - **Cancel evaluation**: 활성 상태일 때
  - **Retry evaluation**(실행 실패) / **Retry submission**(제출 실패): stage와 평가가 모두 `FAILED`이고, 최신 attempt가 제출 실패이거나 Slurm job이 실패 상태이며, 다른 attempt가 활성이 아닐 때만
  - **Re-read result**: stage와 평가가 `FAILED`인데 최신 Slurm attempt는 `SUCCEEDED`이고 `result_path`가 있을 때(`POST /api/evaluations/{id}/reread-result`, `pipeline_api.py:9408-9445, 12203`). 다시 돌리지 않고 이미 있는 `result.json`을 읽어 들인다.
  - `CANCELLED` 평가는 retry할 수 없다(retry는 `FAILED`만, `pipeline_api.py:1997-2004`). 새 평가로 제출한다.
- suite(Evaluations → **Suites**): `dexverse_recorded` tasks-v2(task 4개), `dexverse_training_episode` v1, `libero_*`, `dexmimicgen`, `dexjoco`, `get_zero`, `groot_gr1_tabletop` v3, `groot_gr1_isaaclab_evaltasks` v2, `custom`(manual argv 필요, 추정)

---

## 7. 노트별 실행 계획

- 대상: 아직 실행하지 않았거나 일부만 실행한 노트 6개
- 순서: 폴더별. `egoisim`의 HAT 계열(7-1~7-3) 다음 `warp-extension`(7-4~7-6). 남은 Skynet 작업량 순서가 아니다. 남은 작업은 7-1(평가 제출)이 가장 많고, 7-4·7-6은 준비 결정이 먼저이며, 7-2·7-3은 노트 정정과 기록만 남았다.
- 노트 위치:

  | 폴더 | id |
  |---|---|
  | `egoisim` | `af3704d7-92e6-45d6-b25d-87831065a5b1` |
  | `warp-extension` | `5757129d-0eeb-4d61-bfff-8d30cd69a4ae` |
  | `warp-extension-archive` | `4f16f6ae-3abf-4ff7-8eb5-ce060de501b8` |

### 7-1. HAT baseline + 일곱 손 zero-shot

- **노트:** `2026-09-21-hat-zero-shot-experiment-notes`("수정 실험 계획서: HAT baseline과 일곱 손 zero-shot 일반화"), 폴더 `egoisim`, 상태 줄 없음
- **질문:** 총 데이터와 학습량을 고정했을 때, 학습 손 수 k(1~6)를 늘리면 학습에 없던 손의 PickCube 성공률이 오르는가
- **원본:**
  - HAT `hat_linear`(ACT/CVAE + DINOv2), loss L1 + 10·KL + 2·EEF, AdamW
  - 코드: `https://github.com/RogerQi/human-policy` @ `2d9d73cc5a3859094ef705f35b8f2faecfc2bc4f`
- **Skynet:**
  - 입력: scene RGB 3개(320×240)와 128-D HAT state
  - 출력: dex-retargeting 0.4.6으로 각 손 native controller에 보낸다.
  - 정규화 통계는 source frame에서만 계산한다.
  - 어댑터 v9로 고정했고 DB 최신은 v12다.
  - 학습: run은 preset `custom`으로 제출되었다(`native.config.training_preset`, 210개 모두). 값은 지금 manifest에 있는 `hat-seven-hands-main/v1`(`hat_manifest.py:87-93`)과 같다: 8,000 updates, batch 16, LR 1e-4, 30 Hz, chunk 30, 고유 source frame 1,800, `hand_balanced`, 마지막 checkpoint만 쓴다. preset 이름으로 run을 찾으면 나오지 않는다.

Skynet에 있는 것과 없는 것:

| 항목 | 상태 |
|---|---|
| 학습 | 210/210 `SUCCEEDED`. 손 subset 70개 × seed 1701/1702/1703 |
| 데이터 | HAT 데이터셋 7개 모두 등록됨(§4-6) |
| 평가 suite | `dexverse_recorded` tasks-v2, task `Dexverse-PickCube-v0` |
| evaluator profile | `isaacsim-5.1.0_isaaclab-2.3.2_py311`(attestation 있음, 9/20) |
| 평가 칸(cell) | 계획 735칸 중 101칸 `SUCCEEDED`. Re-read 1칸을 반영하면 102칸 |
| 평가 행(row) | hat-main 평가 행은 829개: `SUCCEEDED` 104(이 중 3개는 seed `[42]` 대조 평가로 칸이 아니다: `543db4fc` 18/20, `cf0abc48` 12/20, `f26c1390` 3/20), `CANCELLED` 724, `FAILED` 1(Re-read 가능, 아래). 실행 중인 것은 없다 |

- `CANCELLED` 724행 중 722행은 9/24 일괄 취소다. 9/23 20:19~9/24에 만든, 그때 열려 있던 칸 전부였다(k1 124, k2 315, k3 84, k4 62, k5 116, k6 21). 그중 k1·k6 칸 다수는 뒤에 새 행으로 끝냈다. 나머지 2행은 9/29 megazord 대체 제출 `6c48dd81`(k6-8d62fa4ab2a7-s1701), `47bc7eaf`(k6-8766ed294d91-s1701)이다(`2026-09-29-research-workstreams-status` 35행).
- 행 수와 칸 수를 섞지 않는다. Evaluations → Runs를 `SUCCEEDED`로 거르면 hat-main 밖의 행 9개(egoverse, HAT 개발, DP, DINO ablation)와 seed `[42]` 행 3개도 섞인다.

k별 남은 평가 칸(DB 읽기 전용 조회, 실험별 SUCCEEDED 평가 수로 계산):

| k | 칸(학습 run × 대상 손) | 완료 | 남음 |
|---|---|---|---|
| 1 | 126 | 83 | 43 |
| 2 | 315 | 0 | 315 |
| 3 | 84 | 0 | 84 |
| 4 | 63 | 1 | 62 |
| 5 | 126 | 10 | 116 |
| 6 | 21 | 7(Re-read 뒤 8) | 14(Re-read 뒤 13) |
| 합계 | 735 | 101(Re-read 뒤 102) | 634(Re-read 뒤 633) |

남은 칸 목록(학습 손은 각 실험의 데이터셋 이름으로 확인):

| 그룹 | 남은 칸 |
|---|---|
| k1 WUJI1(`hat-main-k1-477f637d84c9-s170{1,2,3}`) | 18(seed마다 6) |
| k1 WUJI2(`hat-main-k1-369871ddd6f3-s170{1,2,3}`) | 17(s1701 5, s1702 6, s1703 6) |
| k1 SHARPA(`hat-main-k1-8135f723a579-s170{2,3}`) | 8(s1702 2, s1703 6) |
| k6 s1701 | 0. 대상 SHARPA 칸(`hat-main-k6-8766ed294d91-s1701`, 평가 `0e65e0e9`)은 Re-read만 하면 된다 |
| k6 s1702 | 6(대상 WUJI2 칸 `a16be498e51d-s1702`만 끝남) |
| k6 s1703 | 7 |
| k2 / k3 / k4 / k5 | 315 / 84 / 62 / 116 |

- 이미 돈 칸 하나: 평가 `0e65e0e9-3b92-4b1f-8f7b-6e0efb088afa`(run `8a86b30e…`, `hat-main-k6-8766ed294d91-s1701`, 대상 SHARPA)는 `FAILED`, 진행 97/100이다.
  - 유일한 attempt는 sky2 Slurm 3980862, `SUCCEEDED`/`COMPLETED`, exit `0:0`(grom)이다. 제출 외 event는 `EVALUATION_RESULT_INVALID` `{"error":"sky2: Timeout, server sky2.cc.gatech.edu not responding."}` 하나다.
  - sky1의 `/coc/flash2/ycho420/skynet/eval/runs/cf899265-4d37-4fcb-9ef2-29163960d217/result.json`에 episode 100개, 성공 91개가 있다. 노트(`2026-09-29-research-workstreams-status` 35행 "SHARPA 원본3980862는91/100")와 같다. DB의 episode 행은 성공 88/100으로 일부만 들어가 있다.
  - Re-read 조건을 모두 만족한다(`pipeline_api.py:2006-2011`). Retry evaluation은 최신 attempt가 성공이라 켜지지 않는다.
- k2~k5는 사용자 재개 전까지 보류 중이다(`2026-09-29-research-workstreams-status` 31행 "k2~k5는 사용자 재개 전까지 held 상태다", `2026-09-30-hat-k1-vs-k6-analysis` 17행).

순서:

1. owner가 SHARPA s1701 칸을 Re-read한다. Evaluations → **Runs** → `0e65e0e9…` 행 → **Results** → **Re-read result**(`POST /api/evaluations/0e65e0e9-3b92-4b1f-8f7b-6e0efb088afa/reread-result`, 버튼 `app.js:13849-13850`).
   - DB에 쓰는 동작이므로 owner가 한다. 아무것도 다시 돌리지 않고 기존 `result.json`(91/100)을 읽어 들인다.
   - 이 칸은 다시 제출하지 않는다. 그 뒤 k6는 8칸 완료 / 13칸 남음, 전체는 102 / 633이다.
2. owner가 범위를 정한다.
   - k1 43칸 + k6 13칸(56칸)만 할지
   - k2~k5 보류를 풀어 633칸 전부 할지
3. 남은 칸은 위 "남은 칸 목록"을 쓴다.
   - 칸 하나 = (학습 run, 그 run의 학습 손에 없는 대상 손)이다.
   - 브라우저 목록에는 대상 손이 나오지 않는다(§6-6). 칸을 다시 맞춰 볼 때는 episode viewer의 `robot` 값, 노트의 `hat-main-2026-09-21/dispatch-state.json`, 또는 읽기 전용 DB 조회를 쓴다. k1 run은 대상이 6개라 목록만으로는 구분되지 않는다.
4. 칸마다 브라우저로 평가를 제출한다(AGENTS.md).
   1. Experiments → Runs에서 해당 `hat-main-k<k>-…-s<seed>` run → **View** → **Start evaluation**
   2. **Suite**: `dexverse_recorded`
   3. **Evaluation dataset**: 대상 손의 "· HAT · 51 episodes" 버전. **Unseen hand**는 켠다.
   4. **Tasks**: `Dexverse-PickCube-v0`만 고른다. suite에는 task가 4개 있으니 Select all을 쓰지 않는다.
   5. **Seeds**: `1007,1008,1009,1010,1011`, **Episodes / task**: `20`(100 episodes)
   6. **GPU type**(A40 또는 L40S), **Parallel jobs**(1~8), **Queue policy**를 고른다.
      - CPU는 worker마다 8로 고정된다.
      - L40S는 `grom`, A40은 `megazord`에서 돈다.
   7. **Submit evaluation**을 누른다.
5. 실행 관리:
   - 앱은 run 하나의 여러 대상 손 평가를 동시에 받는다. 새 평가를 막는 것은 그 run의 활성 학습뿐이다(`pipeline_api.py:900-909`, §6-5).
   - HAT 노트의 "평가 job 1개" 운영 한도(노트 81행, 112행 "학습2개/…")는 연구 운영 규칙이다. 앱 제약이 아니다. 유지할지는 owner가 정한다.
   - overcap에서 선점되면 ledger로 이어서 돈다. Auto-resume from episode ledger를 켜 둔다.
   - `CANCELLED` 평가 행은 retry할 수 없다(retry는 `FAILED`만, `pipeline_api.py:1997-2004`). 새 평가로 제출한다.
6. 완료 판정: AGENTS.md 기준으로 Slurm id를 받은 것만으로는 끝이 아니다. 고른 범위의 평가가 모두 `SUCCEEDED`여야 연구가 끝난다.
7. 결과를 기록한다.
   - 노트는 735개가 모두 끝난 뒤에만 `final-results.md/json`을 만든다고 정했다.
   - 범위를 줄이면 그 결정을 노트 첫머리와 본문에 먼저 적는다.

owner가 정할 것:

- k2~k5 보류를 풀지
- 노트의 "평가 job 1개" 운영 한도를 유지할지
- 노트의 추적 파일(`hat-main-2026-09-21`의 `queue-all-status.json`, `dispatch-state.json`)을 어떻게 할지. 이 파일로 개인 배치 스크립트를 되살리지 않고 브라우저로 제출한다.
- 어댑터 버전: 남은 평가도 학습 run에 고정된 v9 manifest로 돈다(spec에 박혀 있음). 새 학습을 추가하면 v12와 섞이지 않게 한다.
- 이 노트(egoisim)에 `- **상태:** 진행` 첫머리를 붙일지(§8)

### 7-2. HAT k1 vs k6 분석의 후속

- **노트:** `2026-09-30-hat-k1-vs-k6-analysis`("2-a·2-b. HAT k=1 vs k=6 비교 분석 (88개 평가)"), 폴더 `egoisim`
- 노트가 적은 후속 세 가지:
  1. 데이터 양 통제: k=6을 총 51개(손당 8~9개)로 줄여 학습한 뒤 k=1과 비교
  2. k=6의 나머지 seed(1702, 1703)와 대상 손 Inspire 평가
  3. k=1의 학습 손 WUJI1·WUJI2 평가 마무리

**전제 오류(확인함):** 노트는 결론 3과 16행에서 "k=6은 학습 데이터가 6배(306개 vs 51개 시연)"라고 썼다. 실제로는 k=1과 k=6이 같은 양의 데이터로 학습했다.

- 두 run의 학습 stdout data summary(sky1, 읽기 전용):
  - `/coc/flash2/ycho420/skynet/logs/hat-main-k1-369871ddd6f3-s1701-train-3891325.out`: `per_hand` `skynet_wuji_2_right` 하나, `unique_source_frames` 1800, windows 756, episodes 36
  - `/coc/flash2/ycho420/skynet/logs/hat-main-k6-85e328d08ea4-s1701-train-3895306.out`: 손 6개 각각 300 frame, episodes 5/7/6/7/7/6(합 38), windows 합 698
- 설계부터 그렇다. HAT 노트 34행 "source 집합마다 고유 소비 원본 프레임 1,800개; 손당 1800/k", 92행. 두 run의 `native.config`에 `unique_source_frames: 1800`, `mixing_policy: hand_balanced`가 있다.
- `apply_frame_budget`(`skynet_app/adapters/unidex_subset.py:49-121`)는 합집합이 예산과 정확히 같지 않으면 "Source subset does not have the requested unique source-frame union"으로 실패한다(120-121).
- 노트가 근거로 든 것은 `data.bundle.assignments` 개수(1 vs 6)와 데이터셋 manifest의 `episodes` 51뿐이다.

후속 항목의 지금 상태:

1. 후속 1(데이터 양 통제): 필요 없다. k=1과 k=6은 이미 고유 frame 1,800개, 비슷한 에피소드 수(36 vs 38)로 학습했다. 새 데이터 통제 실험은 만들지 않는다.
2. 후속 2(k=6 나머지 seed와 대상 Inspire):
   - 대상 Inspire는 이미 끝났다. 평가 `94fa2547`(`hat-main-k6-002d648a277d-s1701`, Slurm 3989055) `SUCCEEDED` 46/100. 노트 68행이 대기 중으로 적은 바로 그 job이다.
   - 노트 "진행 중 (2026-09-30)"의 나머지 둘도 끝났고 DB에 들어왔다(`EVALUATION_IMPORTED`): 3991363 → 평가 `3d7ff10a` 69/100(k6 `a16be498e51d-s1702`, 대상 WUJI2), 3991548 → 평가 `f13ec9dd` 19/100(k1 SHARPA s1702, 대상 LEAP).
   - 남은 것: k6 seed 1702 6칸(대상 WUJI2 칸만 끝남), seed 1703 7칸, 그리고 SHARPA s1701 Re-read(§7-1 1단계).
3. 후속 3(k=1 학습 손 WUJI1·WUJI2): WUJI1 subset `477f637d84c9` 18칸 전부, WUJI2 subset `369871ddd6f3` 17칸(s1701 5, s1702 6, s1703 6)이 남았다.
   - k1의 나머지 열린 칸 8개(SHARPA subset `8135f723a579`: s1702 2, s1703 6)는 노트의 후속 항목에 없다.
4. 평가는 별도 실험 없이 §7-1 순서로 제출한다.

owner가 할 것:

- 노트를 고친다: 결론 3("306개 vs 51개 시연, 데이터 6배")과 16행 학습 조건, 후속 1, "진행 중" 절. 노트 쓰기는 §8의 API나 브라우저로 한다.
- 후속 1의 "총 51개" 정의 질문은 다시 쓴다. 에피소드 수가 이미 거의 같다(36 vs 38).

### 7-3. 연구 진행 현황 (Lightwheel · HAT · DP · DexVerse)

- **노트:** `2026-09-29-research-workstreams-status`, 폴더 `egoisim`
- 남은 항목과 지금 상태:

| 항목 | 노트 기록 | DB 현재 | 할 일 |
|---|---|---|---|
| 2-a / 2-b HAT 평가 | 미완 | §7-1과 같은 수 | §7-1 |
| 3-b DP vs HAT(WUJI2, 같은 손) | DP 평가 대기 | DP 평가 `f93e7eaa-9d46-4648-96a3-5fbdfdb3023b` `SUCCEEDED` **9/100**(2026-09-30T05:20Z). HAT `31715ec9…` **93/100** | 새 실험 없음. 결과를 노트에 기록 |
| 3-c DINO ablation | "미제출" | `hat-wuji2-pickcube-51-fit-s42-resnet18`(평가 `5beb373d…`) **92/100**, `…-dinov2`(`e140c9cd…`) **99/100**, 둘 다 2026-10-01 완료 | 새 실험 없음. 결과를 아래 조건과 함께 노트에 기록 |
| 1-b Lightwheel 수집을 Skynet에 통합 | 구현 보류(가능성만 확인) | 변화 없음 | 실험이 아니라 앱 기능이다. owner가 착수를 정한다 |

- 3-b 조건:
  - 원본: DP `real-stanford/diffusion_policy@5ba07ac6`(U-Net, ResNet18, DDPM100, EMA)
  - Skynet: 어댑터 `diffusion-policy` v2, preset `dp-hat-fit-comparison/v1`, 3,000 updates, batch 16, seed 42, 60 Hz, chunk 50, RGB 3개 + WUJI2 joint 26D
  - HAT 비교 기준값이 둘이다: 93/100(`31715ec9`, run `f91b27ed`, 어댑터 v4, preset `hat-wuji2-fit/v1`)과 99/100(3-c의 v11 DINOv2 재실행 `e140c9cd`). 같은 명목상 fit recipe다. 노트에는 어느 값과 비교했는지 적는다.
- 3-c 조건:
  - 원본: HAT repo의 공식 config 두 개. `act_resnet`(`hdt/configs/models/act_resnet.yaml`, ResNet18)과 `hat_linear`(DINOv2)
  - Skynet: 어댑터 `human-policy-hat` v11, resnet18 run은 preset `hat-fit-resnet/v1`(`act_resnet`), dinov2 run은 `hat-fit/v1`(`hat_linear`)(`hat_manifest.py:78-86`). 둘 다 3,000 updates, batch 16, seed 42, chunk 50, `window_policy` pad, WUJI2, 60 Hz(원본 주파수)
- 순서:
  1. Evaluations → Runs → 위 평가의 **Results**에서 성공률과 episode 수를 확인한다.
  2. §8 규칙으로 결과 노트에 적는다(`### 2026-09-30 DP vs HAT (WUJI2)` 등).
  3. 비교는 같은 hand, 60 Hz, seed 1007~1011 × 20 조건일 때만 한다. 조건이 같은지 각 평가의 suite/seed를 Results에서 확인한다.

### 7-4. S3 계획 · EgoHumanoid 기준선 (RGB 재투영)

- **노트:** `2026-10-06-s3-plan-egohumanoid-baseline`, 폴더 `warp-extension`, 상태 `진행 (10/5 착수)`
- **질문:** WARP++가 점군을 쓰는 근거가 있는가(시점만 맞춘 RGB와 비교)
- **원본:**
  - 논문: EgoHumanoid, arXiv 2602.10106(RSS 2026)
  - 코드: `https://github.com/OpenDriveLab/EgoHumanoid` `data_alignment/view_alignment/`. revision은 고정하지 않았다.
  - 순서:
    1. MoGe 단안 깊이로 3D로 올린다.
    2. 고정된 카메라 이동으로 옮긴다(README 예시 down 0.07, 샘플마다 ±0.02, 단위 미확인).
    3. 2D로 다시 투영하고 빈 곳을 Stable Diffusion inpainting으로 채운다.
    4. 바뀐 이미지로 π0.5를 학습한다.
- **비교 대상(Chuye WARP++ 설정, Skynet 설정 아님):**
  - 모델: DP3 PointNet encoder + HPT backbone
  - 점군: head camera 기준 global(깊이 0.25~2.0 m), 오른쪽 EEF 1.5 m crop local
  - 양쪽 EEF에서 0.30 m 안의 점은 색만 회색으로 바꾸고 기하는 그대로 둔다(`2026-10-06-00-index-warp` 22행, S4 노트 14행).
  - 학습 코드: Chuye fork `/coc/flash7/czhang883/Documents/EgoVerse`. `git log`가 "dubious ownership"으로 실패해 revision을 확인하지 못했다.
  - 학습 sbatch: `/coc/flash7/czhang883/warp-data/pipelines/chuye_basket/train_dp3cbasket_v2.sbatch`(sky1 확인). fork 디렉터리로 `cd`한 뒤 `source emimic/bin/activate`, `srun python egomimic/trainHydra.py --config-name=experiments/wholebody_image/wb_dp3cbasket name=RBY1_dp3c_basket_v2 …`를 돈다. `gpu:a40:1`, `cpus-per-task 8`, overcap, `--requeue`, 30 h. fork 안의 `pipelines/chuye_basket/`는 없다.

Skynet에 있는 것과 없는 것:

| 항목 | 상태 |
|---|---|
| EgoHumanoid checkout/환경 | 없음. sky1 `/coc/flash7/ycho420/repos`에 없다 |
| view-alignment 변환 단계 | 없음 |
| 어댑터 | 전용 어댑터는 없다. `generic`(v145)으로 감쌀 수 있다. π0.5 어댑터 둘은 RBY1 재투영 데이터를 지금 그대로는 학습하지 못한다: `openpi`는 LIBERO 스키마 LeRobot(`image`, `wrist_image` RGB, `state[8]`, `actions[7]`)과 `norm_stats.json`만 받는다(`openpi_dataset_bridge.py:32-60`, 도움말 `adapters/__init__.py:3294-3300`). XPolicyLab Pi_05 Native는 XPolicyLab recipe로 준비한 `xpolicylab-native-pi_05/v1`(contract `skynet.xpolicylab-native/v1`)만 받는다(`xpolicy_native_manifest.py:61-77`). `egoverse-hpt`는 점군 입력이 없다 |
| 데이터 | 둘 다 Skynet에 등록되지 않았다(sky1에서 경로 확인). 아래 표 참고 |
| 평가 | S3 노트는 "고정된 장면에서 같은 시기에 평가한다"(50행)만 적었다. 평가 조건은 "정할 것"(38행)으로 남아 있다. 실제 RBY1에서 한다는 것은 목차 노트 20행(로봇 정책)과 S2 노트 49행(1305호 조명·방 배치로 평가가 흔들림)에서 읽은 추론이다(추정). Skynet simulator suite는 해당 없음. episode 수는 정해지지 않았다 |
| 추적 | Notion 티켓 `https://app.notion.com/p/3f05677e5d1f8053b7ead703b6594100`. Chuye의 이전 basket run은 W&B project `sew_policy`(노트 `2026-09-25-warp-basket-collector-analysis` 60행, run `RBY1_dp3c_basket_v2_basket60_v2` 등) |

| 데이터 | 내용 |
|---|---|
| `/coc/flash7/czhang883/warp-data/datasets/dp3c_basket60_v2/releases/r1` | LeRobot v2.0 parquet, `robot_type rby1`, 53 episodes, 18,909 frames, `fps 10`, train/valid split |
| `/coc/flash7/czhang883/warp-data/collections/chuye_basket/processed/r1/rgbd/toy_rgbd_basket.hdf5` | 같은 시연의 RGB-D, 1,616,480,790 bytes |

- 주의(대체된 archive 노트 `2026-09-25-chuye-research-experiment-plan` 18행, 폴더 `warp-extension-archive`, 상태 대체됨): 사람 basket row는 센서 기준 약 60 Hz인데 metadata와 학습 config는 10 fps로 다룬다. release `meta/info.json`도 `fps 10`이다.
- release `meta/info.json`(sky1): `codebase_version v2.0`, `total_videos 0`, RGB feature 없음. feature는 `obs.aria_pcdc[6144]`, `obs.aria_pcdc_local[6144]`, `obs.eef_pose_glass[9]`, `obs.robot0_joint_pos[26]`, `actions.joint[49]` 등이다.
- 읽기 권한: release `r1`은 `drwxr-sr-x`, `toy_rgbd_basket.hdf5`는 `-rw-r--r--`이고 그룹은 `coc-skynet-access`다. ycho420이 sky1에서 둘 다 읽었다.

순서:

0. Skynet 밖에서 먼저 끝낼 것(노트의 "다음"과 "정할 것"):
   - 논문 읽고 코드 돌려 보기
   - Notion 티켓에 WARP++ 목표·제약과 S3·S4 평가 조건 쓰기
   - Chuye 검토
   - 정할 것: 정책 구조는 그대로 두고 관측만 바꿀지, 깊이 추정 방법, 고정된 평가 조건
1. EgoHumanoid commit을 40자 sha로 정해 노트에 `(원본)`으로 적는다. 그 commit이 원격 branch의 최신 50개 안에 있어야 **Commit**에서 고를 수 있다(§5-1).
2. owner가 클러스터를 준비한다(§2-2).
   - generic job의 코드는 **Repository**와 **Commit**으로 job cache에 clone된다(`slurm.py:1474, 1606-1634`). 공유 checkout은 필요 없다. profile을 검증하거나 Convert·evaluator에 쓸 때만 필요하다(§2-1).
   - 환경: MoGe와 SD inpainting 의존성을 버전 고정으로 설치한 venv(`bin/activate`가 있어야 한다, §2-3)
   - 재현 기록이 필요하면 `skynet.json`에 학습 profile을 추가하고 앱을 재시작한다(§2-3).
3. view alignment를 `generic` job으로 돌린다(§3-3).
   1. Submit → **Algorithm / adapter** "Custom structured command"
   2. **Repository** `https://github.com/OpenDriveLab/EgoHumanoid`, **Branch**, **Commit** `<sha>`
   3. **Runtime**: Existing environment (manual) + venv 경로, 또는 **Named runtime profile**
   4. **Adapter-declared inputs** → **Training command**: 한 줄에 토큰 하나. 스크립트는 checkout 기준 상대 경로(예: `data_alignment/view_alignment/...`)나 `{{SKYNET_SOURCE_DIR}}`로 쓴다. 실행할 스크립트와 인자는 upstream README에서 정한다. 이 가이드에서 확인하지 않았다.
   5. 입력 데이터와 출력만 절대 경로로 쓴다. `--chdir`로 손 checkout에 들어가지 않는다(Lightwheel 선례는 따르지 않는다, §3-3).
   6. Auto-resume를 끈다. **GPU allocation**은 1개이고 CPU는 자동 8이다.
   7. **Preview sbatch**, **Create and submit**
   - 입력이 다른 사용자 디렉터리(`czhang883`)지만 그룹 `coc-skynet-access` 읽기 권한이 있고 ycho420으로 읽힌다(위 데이터 표 아래).
4. 정책 학습: 0단계의 결정에 따라 갈린다.
   - **(a) WARP++ 정책 유지:** Chuye fork의 commit을 원격 branch에 push해 40자 sha로 고정해야 job cache가 clone할 수 있다. 그 sha가 branch의 최신 50개 안에 있을 때 고른다(§5-1). 그다음 `generic` job으로 학습한다. 진입점은 `egomimic/trainHydra.py --config-name=experiments/wholebody_image/wb_dp3cbasket …`(위 sbatch)이고, fork의 `emimic` venv 대신 쓸 환경을 따로 정해야 한다.
   - **(b) π0.5(EgoHumanoid 원본 방식):** 지금 있는 π0.5 어댑터로는 안 된다. `openpi`는 LIBERO 스키마 LeRobot과 `norm_stats.json`만, Pi_05 Native는 XPolicyLab-native 준비 데이터만 받는다(위 표). **Add files**로 등록해도 고를 수도 학습할 수도 없다. 새 bridge/어댑터(코드 변경과 재시작, §3-4)를 만들거나, 사용자의 π0.5 trainer를 `generic` job으로 돌려야 한다.
   - **(c) 정식 경로:** 외부 데이터셋 등록·검증 기능을 코드로 먼저 만든다(§4-1).
5. 평가는 Skynet 밖에서 한다. 실제 RBY1에서 할 것으로 보이지만 평가 조건은 아직 정하지 않았다(추정). 성공 수/시행 수를 노트 `## 2. 결과`에 적는다.

owner/Chuye가 정할 것:

- 0단계 세 가지
- 4단계 (a)/(b)/(c)
- 60 Hz와 10 fps 불일치를 재투영 데이터에서 어떻게 다룰지
- 평가 episode 수

### 7-5. S4 계획 · 점군 inpainting

- **노트:** `2026-10-06-s4-plan-point-cloud-inpainting`, 폴더 `warp-extension`, 상태 `진행 (방법 탐색 전)`
- **질문:** 사람 손이 남은 점군이 성능을 깎는가
  - 해석: inpainting이 회색 처리보다 높으면 손 모양 차이가 성능을 깎고 있었다는 뜻이다.
- **원본:** 없음. 정해진 방법이 없어 논문부터 읽는다(Chuye, 10/5).
- **지금 처리:** 양쪽 EEF에서 0.30 m 안의 점은 색만 회색으로 바꾸고 기하는 그대로 둔다(S4 노트 14행, S3와 같은 WARP++ 설정).

Skynet에 있는 것과 없는 것:

| 항목 | 상태 |
|---|---|
| 방법/코드 | 없음 |
| 어댑터 | 없음 |
| 데이터 | 노트에 없음. S3와 같은 basket 데이터로 보인다(추정) |
| 학습/평가 설정 | 없음 |
| 추적 | Notion 티켓 `https://app.notion.com/p/3f05677e5d1f80ff9532d079b5fd8500` |

순서:

1. Skynet 밖: 관련 논문을 읽고 후보 방법을 정리해 Chuye 검토를 받는다.
2. 노트 "정할 것"을 정한다.
   - 사람 손 점을 지우기만 할지, 로봇 손 모델로 채울지
   - 안경의 어떤 정보를 쓸지
   - 가려진 물체를 어떻게 처리할지
3. 방법이 정해지면 S3 3~5단계와 같은 경로를 쓴다.
   - inpainting 전처리 → 학습 → 실물 평가(평가 조건 미정)
   - 외부 코드가 있으면 §2로 가져오고 `generic` job으로 돌린다.
4. 비교 조건(회색 처리 vs inpainting)은 같은 task, 같은 평가 조건으로 맞춘다. S2 B2가 이 성공률을 쓴다.

정할 것: 위 2단계 세 가지, 데이터 출처, 평가 episode 수

### 7-6. S2 정리의 다음 과제 B1~B4

- **노트:** `2026-10-05-warp`("S2 정리 · 행동 분포 연구 경과와 다음 과제"), 폴더 `warp-extension`, 상태 `진행 (Chuye가 10/5 우선순위를 조금 낮춤)`
- **지금까지:** S2는 개인 sbatch로 Skynet 밖에서 돌렸다. Skynet experiment row는 없다.
  - 파일럿: Slurm 3990523, 3996091
  - coffee 통제: Slurm 3993377~3993390, `train_coffee.sbatch`
  - 데이터 `/coc/flash7/ycho420/datasets/warp-coffee-variants`, run `/coc/flash7/ycho420/runs/warp-coffee`
  - 코드: Zhenyang EgoVerse `94b9f7f`(근거는 `2026-10-01-warp-dexmimicgen-coffee-controlled` 41행과 디렉터리 이름뿐). 로컬 사본 `/coc/flash7/ycho420/repos/EgoVerse-zhenyang-94b9f7f`에는 `.git`이 없다(`git log`가 "not a git repository"로 실패). 따라서 generic job의 **Repository**와 **Commit**에 넣을 원격 URL과 40자 sha는 Zhenyang 저장소에서 받아야 한다.
  - 설정: HPT + FMPolicy, 32행 chunk, batch 32, AdamW 1e-4, bf16, epoch 999 ≈ 100k updates, seed 1·2, A40 1개 ≈ 12 h
- **노트의 방침:** B는 따로 큰 실험을 하지 않는다. S3·S4와 WARP++ ablation 결과 옆에 지표를 붙인다(Skynet 제안).

| 과제 | 막는 것 | Skynet에서 할 일 |
|---|---|---|
| B1. 학습에 안 쓴 시작 조건으로 기존 coffee 정책 rollout(조건당 약 100회, 노트 추정) | 새 시작 조건 정의. 정책이 Skynet 밖에서 학습되어 Skynet 학습 run이 없다 | Evaluations 폼은 쓸 수 없다. **Training run ID**가 필수이고("Training run ID is required.", `pipeline_api.py:10027-10028`), checkpoint override도 그 run의 `AVAILABLE` checkpoint 중 하나여야 한다(`pipeline_api.py:10035-10050`). 외부 checkpoint는 평가할 수 없다. Zhenyang rollout client + `serve_policy`(flow step 10)를 `generic` job으로 감싸 돌린다. Zhenyang fork의 원격 URL과 40자 commit이 필요하다. DB의 `dexmimicgen` 어댑터(v107)와 suite는 S2가 쓰지 않았다 |
| B2. S3·S4 ablation 성공률 옆에 지표 붙이기 | S3·S4 결과. 같은 task, 고정 평가, 설정 3개 이상일 때만 | 인코더, PCA 범위, 상태 표현, 구간 계산을 성공률을 보기 전에 노트에 고정한다. 계산 job은 `generic` |
| B3. WARP++ 실시연에서 MINK vs WARP 카메라 자세 일관성 | Chuye·Youngwoong basket 데이터의 MINK·WARP 변환본(Chuye가 10/6에 경로 전달 예정, Notion "Action analysis additional data", S2 노트 93행) | 데이터를 받은 뒤 `generic` job |
| B4. 공개 데이터에서 MI로 고른 시연 vs 무작위(DemInf 방식) | 공개 데이터셋을 정하지 않았다. 후보: bones-seed 데이터셋(S2 노트 93행, 경로는 Chuye가 10/6 전달 예정) | 데이터셋이 HF에 있으면 §4-3 import(Datasets 탭). 학습 어댑터는 데이터 format에 따라 정한다 |

- 기존 데이터 출처: `/coc/flash7/zhenyang/datasets/first50_base_fixed/<task>_<method>_first50.hdf5`, `/coc/flash7/zhenyang/EgoVerse/...`(LeRobot 사본과 checkpoint)
- 앞으로는 개인 sbatch를 쓰지 않는다. 같은 작업을 Skynet `generic` job으로 옮긴다(AGENTS.md "no private execution entry points").

정할 것:

- B 착수 시점(우선순위 낮아짐)
- B1의 새 시작 조건과 rollout 수
- B4의 공개 데이터셋

### 7-7. 돌리지 않는 노트 (대체됨/보류/owner 결정)

| 노트 | 상태 | 이유 |
|---|---|---|
| `2026-09-25-chuye-research-experiment-plan` §10 A~F | 대체됨 | S2/S3/S4로 대체. Skynet run 없음 |
| `2026-09-15-unseen-embodiment-experiment-notes-a105abbc`, `readme-4a8e116b`, `intent-8d809234`(UniDex) | 대체됨/중단 | UniDex 어댑터 제거(`12234f0`), run은 DB에 없음. HAT(§7-1)로 대체 |
| `hand-adapters-implementation-plan-29ba12b9`(UniDex·OPFA 어댑터 설계) | 대체됨 | `faas-hpt`, `opfa-hpt`, `opfa-native`는 만들지 않았다. `unidex-native`는 만든 뒤 제거했다 |
| `2026-09-25-warp-basket-collector-analysis` §7, `2026-09-26-warp-action-distribution-plan` §12 | 보류 | 노트에서 보류로 표시 |
| `2026-09-26-warp-action-distribution-results` §9 후속(오른손 예측·명령·실측 저장, C 4개 해제 tail 복원, Y 4개 action index 15 확인) | 보류 | 노트 6행 "9/29 이후 S2로 넘어가며 보류" |
| `2026-09-30-warp-dexmimicgen-metrics-pilot` §4 3항 "coffee 실패 분석" | owner 결정 필요 | 첫머리 "다음"에 남아 있고 하지 않았다. `2026-10-05-warp` 98행 "멈추는 것"은 DexMimicGen 통제 실험의 추가 설계를 멈추지만 이 분석을 명시적으로 닫지는 않는다 |
| `2026-09-20-unidex-current-status-c0708a0d`, `validation-10d22cc1`, `skynet-integration-audit-643a4292`(`faas-hpt`/`opfa-hpt`를 권고) | 대체됨/중단 | UniDex·OPFA 시기 계획. UniDex 어댑터 제거(`12234f0`), HAT로 대체 |
| `2026-10-03-lightwheel-collector-pr-fixes` | 실험 아님. owner 확인 필요 | 노트 9행의 남은 것: push와 PR 생성, 종료 코드 수정의 GPU 확인, G1 정지 대기 문제. 노트는 세 로봇 수집이 됐다고 적지만, DB의 PR 수집·검증 run(Slurm 3994437, 3994439, 3995516, 3995519, 3995521, experiment `lightwheel-*-pr-20261001*`)은 모두 `FAILED`(exit `1:0`)다 |

---

## 8. 결과 기록

규칙 원문은 `skynet_app/notes_guide.md`다. 화면에서는 Notes의 **Writing guide**, API는 `GET /api/notes/guide`로 읽는다. 새 노트 틀은 `skynet_app/note_template.md`다.

화면(Experiments → **Notes**):

- **New folder**, **New note**(계획 노트 틀이 채워짐)
- **Edit**에서 **Folder**를 바꾸고 **Attachments**로 파일을 올린다.
- 저장 전에 **Check format**을 누른다.
- 폴더로 바로 가는 링크: `/?cluster_view=gpu&experiment_view=notes&note_folder=<folder-id>#experiments`

API(prefix `/api/notes`, `notes.py:417`. 작업 트리의 `notes.py`는 수정 중이라 줄 번호가 바뀔 수 있다):

| 동작 | Route |
|---|---|
| 목록/생성 | `GET`, `POST /api/notes`. POST 본문은 `{title, markdown, folder_id}`이고 `folder_id` 키는 필수다(`null` 가능, `notes.py:68-72`) |
| 규칙 | `GET /api/notes/guide`(`template` 포함) |
| 형식 검사 | `POST /api/notes/check` `{"title": ..., "markdown": ...}`. `problems`가 비면 통과 |
| 폴더 생성 | `POST /api/notes/folders` `{"name": ...}` |
| 읽기 | `GET /api/notes/{id}` |
| 수정/이동 | `PUT /api/notes/{id}`는 노트 전체를 바꾼다. 본문은 `{title, markdown, folder_id, expected_updated_at}` 전부(`NoteUpdate`, `notes.py:84-85, 462-466`). 다른 곳에서 바뀌었으면 409 "This note changed elsewhere. Reload it and try again."(`notes.py:346, 430-431`). 이동도 같은 전체 PUT에 새 `folder_id`를 넣는다 |
| 삭제 | `DELETE /api/notes/{id}`(`notes.py:468`). maintenance API를 거치지 않는다 |
| 내려받기 | `GET /api/notes/{id}/download` |
| 첨부 | `GET`, `PUT`, `DELETE /api/notes/{id}/attachments/{name}`(`notes.py:480-501`). 올리기는 raw request body로 `PUT`하고 크기 제한이 있다 |

- Claude 세션은 쓰기 전에 사용자 확인을 받는다. 노트는 중앙 DB `notes` 테이블에 있으므로 DB를 직접 고치지 않고 이 API나 브라우저로만 쓴다.
- 노트 `2026-09-25-warp-experiment-workflow`의 "`data/notes/legacy/<id>.md` 파일을 고친다"는 지금은 틀렸다. 그 디렉터리는 없다.

형식 요점:

- **폴더:** `<프로젝트>`(진행 중 노트와 목차)와 `<프로젝트>-archive`(완료/대체됨). 이름은 소문자, 숫자, 하이픈. 폴더는 이름 바꾸기와 삭제가 없다.
- **제목:** `<번호> <종류> · <내용>`
  - 종류: 계획/결과/정리/참고/운영
  - 목차는 `00 목차 · <프로젝트> 연구 노트`이고 맨 위에 고정된다.
- **첫머리:** `# <제목>` 다음 다섯 줄, 그다음 메타 줄 `YYYY-MM-DD · 담당 <이름> · <한 줄>`
  ```markdown
  - **상태:** 진행
  - **질문:** ...
  - **결론:** 아직 없음.
  - **다음:** ...
  - **관련:** [00 목차](2026-10-06-00-index-warp.md) · [S3 계획](2026-10-06-s3-plan-egohumanoid-baseline.md)
  ```
  - Status 열은 첫 2,000자 안 첫 `- **상태:** <단어>` 줄의 첫 단어를 읽는다(`notes.py:31`).
  - 상태는 진행/보류/완료/대체됨이다. 완료와 대체됨은 archive 폴더로 옮긴다.
- **결과 본문:** `## 요약`, `## 1. 설정`, `## 2. 결과`, `## 3. 해석`, `## 4. 다음`, `## 5. 진행 기록`
  - 같은 연구의 다음 실험은 새 노트를 만들지 않는다. `## 2. 결과` 아래에 `### YYYY-MM-DD <실험 이름>`으로 붙이고 요약과 첫머리를 고친다.
- **링크:** 노트는 `[글](<note-id>.md)`로 연결한다. `localhost` 주소는 넣지 않는다. 그림은 Attachments에 올리고 `![무엇: 조건](plot.svg)`로 부른다.
- **숫자:** 단위, 표본 수, 구간을 함께 쓴다. 측정하지 않은 값은 `(추정)`이나 `미확인`으로 쓴다.
- **출처:** `(원본)`, `(Skynet 제안)`, `(Chuye, 10/5)`처럼 괄호에 쓴다.
- **바꾼 설정:** "조건"이라고 부른다.
- **문장:** em dash와 문장을 잇는 세미콜론을 쓰지 않는다.

실험 기록에 넣을 Skynet 식별자:

- experiment 이름, run id, 평가 id, Slurm job id
- 어댑터 slug와 버전, 원본 repo@commit, runtime profile id
- 데이터셋 version id, suite와 version, seed 목록, episode 수
- W&B run 링크

채울 빈칸:

- `egoisim` 노트 44개에는 상태 줄이 없다. 목차 노트와 `egoisim-archive` 폴더도 없다. HAT 노트(§7-1)를 고칠 때 첫머리를 붙일지 owner가 정한다.
- §7-3의 3-b와 3-c 결과는 DB에만 있고 어느 노트에도 없다.

---

## 9. 하지 말 것과 주의

AGENTS.md 규칙:

- **GPU 1개당 CPU 8개.** 학습, 평가, 그 밖의 GPU job 모두, GPU 종류와 무관하게 적용한다.
  - 서버가 강제한다(§5-4). manifest나 예전 spec의 `cpus_per_task`를 믿지 않는다.
  - 이미 돌고 있는 job은 건드리지 않는다. 10/5 DB 기준 실행 중인 run과 평가는 없다.
- **HAT는 정식 브라우저 workflow로 제출한다.** 학습과 평가 모두 해당한다. 개인 배치 스크립트나 API 직접 호출로 대량 제출하지 않는다.
- **삭제는 정식 maintenance API로 한다.** maintenance kind 10개(`experiment`, `draft-revision`, `run`, `evaluation`, `adapter`, `suite`, `dataset`, `prepared`, `recording`, `recording-file`, `maintenance_api.py:15`)의 **Delete** 버튼(`data-delete-kind`, `static/maintenance.js`)은 `GET` 미리보기 → preview token을 실은 `DELETE /api/maintenance/history/{kind}/{id}`를 거친다(`maintenance_api.py:66,77`).
  - 노트, 노트 첨부, hand pose는 자기 route로 지운다: `DELETE /api/notes/{id}`(`notes.js:423`), `DELETE /api/notes/{id}/attachments/{name}`(`notes.js:286`), `DELETE /api/hands/{key}/{side}/poses/{id}`(`hands_api.py:81`).
  - HTTP `DELETE /api/adapters/{id}`는 삭제가 아니라 보관이다(§3-2).
  - 개인 실행 경로나 삭제 경로를 만들지 않는다. 예: 개인 sbatch, `import_dexverse_release.py` 수동 실행, DB 직접 수정
- **UniDex는 영구 삭제가 허용되어 있다.** HAT와 공유 자산은 보존한다.
  - 남은 UniDex 데이터셋 7개(`skynet.unidex-pointcloud-faas/v1`, `AVAILABLE`)는 받는 어댑터가 없다. 정리한다면 maintenance kind `dataset`으로 한다.
- **원본과 Skynet을 나눠 쓴다.** 모델, 학습, 추론, 평가, 데이터 설명 모두에 적용한다(§2-6).
- **HAT 연구는 학습과 평가가 모두 끝나야 끝난다.** Slurm id를 받은 것만으로는 끝이 아니다.

그 밖:

- 호스트를 하드코딩하지 않는다.
  - gateway는 설정 `gateways`(`sky1`, `sky2`)와 헤더 **Cluster gateway**를 쓴다.
  - 경로는 `skynet.json` `paths.*`(`work_root`, `repositories`, `shared_repositories`, `environments`, `datasets`, `logs`, `jobs`)를 기준으로 쓴다. 개인 경로(`work_root`, `workspace`, `repositories`, `artifacts`, `logs`, `jobs` 등)는 workspace base path로 rebase된다(`workspace_storage.py:17-48`). 기존 run의 경로는 `runs.run_directory`를 본다.
- 중앙 PostgreSQL을 직접 고치지 않는다. 읽기도 읽기 전용 연결로만 한다. 쓰기는 앱 API/브라우저로 하고, 그것으로 안 되는 행 수정은 owner가 한다.
- job cache 경로 `repos/<name>-<sha8>/<sha>`를 손으로 만들거나 고치지 않는다(exit 65). Repository URL 철자를 앞선 run과 똑같이 쓴다(다르면 cache가 새로 생긴다, §2-1).
- 브랜치 이름이나 짧은 sha로 제출하지 않는다. 40자 commit만 받는다.
- generic argv에서 `--chdir`로 손 checkout에 들어가지 않는다. 기록된 commit과 다른 코드가 돈다(§3-3).
- `xpolicylab-act`, `isaacsim-5.1.0_isaaclab-2.3.2_py311` profile과 queue key `normal`의 이름을 바꾸거나 지우지 않는다(§2-3).
- profile JSON을 고치면 그 profile의 attestation이 stale이 된다. evaluator profile이면 readiness smoke를 다시 돌린다(§2-4).
- `skynet.json`이나 어댑터 코드를 고친 뒤에는 재시작을 잊지 않는다. built-in을 브라우저에서 고치면 그 slug의 코드 seed가 멈춘다.
- 브라우저는 Validation episodes 0%를 막지 않고 "Training will run without validation steps."라고 경고만 한다. 하지만 HAT/DP/ACT Native loader는 validation window를 요구하므로(`require_validation=True`) 이 어댑터들의 데이터는 0%보다 크게 Convert한다(§4-2).
- 평가 폼의 Tasks에서 Select all을 누르면 `dexverse_recorded`의 4개 task가 모두 돈다. HAT 연구는 `Dexverse-PickCube-v0`만 쓴다.
- 비밀값(`config/database-password`, token, W&B key, cookie)을 출력하거나 노트에 쓰지 않는다.

---

## 부록 A. 코드와 다른 문서

| 문서 | 현재 UI/코드 |
|---|---|
| `README.md:86`, `docs/slack-notifications.md:3` "Settings → Slack" | Settings → **Notifications** → Slack |
| `README.md:100` "Experiments → Slurm resources → GPU allocation → Manual" | Experiments → Submit → **5. Resources** → GPU allocation = Manual, **GPUs / node** |
| `README.md:117` "Training Runs → View attempts → Start evaluation" | Experiments → **Runs** → **View** → **Start evaluation** |
| `README.md:121-124` "Rollout videos table" | Evaluations → Runs → **Results** → **Episode results** → **Detail** |
| `README.md:319` 평가 submit endpoint | 별도 route 없음. `POST /api/evaluations`가 생성과 제출을 함께 한다 |
| `README.md:322-324` `PATCH`, `/archive`, `/validations` 어댑터 route | 없음(§3-2 표가 실제 route) |
| `docs/training-contracts.md:71-72` "Training data", "GPU and time" | "2. Training datasets", "5. Resources" |
| `docs/email-workspaces.md:19` "Settings → Cluster storage" | Settings → **Storage** 탭 → Cluster storage |
| 노트 `2026-09-25-warp-experiment-workflow` "`data/notes/legacy/<id>.md`" | 노트는 중앙 DB에 있다. `/api/notes`나 브라우저로 쓴다 |

## 부록 B. 더 읽을 문서

- `README.md`: "UI workflow"(88-130), "Canonical experiment model"(134-183), "API workflow"(314-337), "Real Slurm behavior"(339-372), "Checkpoints, retention, and resume"(474-486), "Evaluation commands and data"(488-498), "Reproducibility capsule"(500-525)
- `docs/training-contracts.md`: preset, 제출 순서, 다중 데이터셋, held-out 손, HAT 원본 vs Skynet
- `docs/training-metrics.md`, `docs/time-limit-recovery.md`, `docs/evaluation-compatibility.md`, `docs/episode-viewer.md`, `docs/policy-data-exports.md`, `docs/live-dexverse.md`, `docs/egoverse-models.md`, `docs/xpolicylab-adapters.md`
