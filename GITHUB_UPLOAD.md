# GitHub 업로드 가이드

## 최신 버전 가져오기 (pull / 로컬 업데이트)

GitHub `origin`에 올라와 있는 최신 버전으로 이 디렉토리를 업데이트할 때 사용합니다.

### 1단계 — 원격 변경분 확인 (fetch)

```bash
cd /sjw_alinlab1/home/jungwook/action_tokenizer

git fetch origin
git log --oneline -5 origin/master            # 원격 최신 커밋 확인
git rev-list --left-right --count origin/master...master   # 좌=원격 앞섬, 우=로컬 앞섬
```

### 2단계 — 로컬에 커밋 안 된 변경분이 없는 경우 (일반)

```bash
git pull --ff-only origin master
```

> `--ff-only`은 깨끗한 fast-forward 업데이트만 허용합니다. 로컬이 원격과 갈라져 있으면 거부되므로 안전합니다.

### 2-1단계 — 로컬에 커밋 안 된 변경분이 있는 경우

`git pull`이 `Your local changes ... would be overwritten` 오류로 막히면, 로컬 변경분을 잠시 보관(stash)한 뒤 pull하고 다시 적용합니다. (변경분은 버려지지 않고 보존됨)

```bash
git stash push -m "before pull"   # 로컬 변경분 임시 보관
git pull --ff-only origin master  # 최신 버전으로 업데이트
git stash pop                     # 보관한 변경분 재적용 (필요시 충돌 해결)
```

> 로컬 변경분이 더 이상 필요 없다면 stash 대신 `git checkout -- <파일>` 또는 `git reset --hard origin/master`로 버릴 수 있습니다. **`reset --hard`는 로컬 변경/커밋을 완전히 삭제하므로 주의하세요.**

### 3단계 — 업데이트 결과 확인

```bash
git log --oneline -3
git status
```

---

## 방법 0: 기존 레포에 현재 버전 다시 푸시 (현재 상황)

이미 `origin`(https://github.com/snu-jungwook/action-tokenizer.git)이 연결되어 있고
`master` 브랜치가 존재합니다. 새로 만들 필요 없이 변경분만 커밋해서 push하면 됩니다.

### 1단계 — 현재 상태 확인

```bash
cd /NHNHOME/data/wook/action-tokenizer

git status          # 변경/추가된 파일 확인
git remote -v       # origin 연결 확인
```

### 2단계 — 변경분 스테이징 및 커밋

```bash
# 모든 변경분(수정 + 신규 파일) 스테이징
git add -A

# 커밋 (메시지는 원하는 내용으로 변경)
git commit -m "Update action tokenizer: v5 LAM modules, smoketests, sbatch scripts"
```

> 일부 파일만 올리고 싶으면 `git add -A` 대신 `git add <파일경로>`로 개별 추가하세요.

### 3단계 — push

```bash
git push origin master
```

> 인증 창이 뜨면 username 입력 후 password 자리에 GitHub Personal Access Token(PAT)을 붙여넣으세요.
> `gh auth login`으로 미리 로그인해두면 인증 과정이 생략됩니다.

### (선택) 강제 덮어쓰기

원격 히스토리를 로컬 기준으로 완전히 덮어써야 하는 경우에만 사용하세요. **원격 커밋이 사라질 수 있습니다.**

```bash
git push --force origin master
```

---

# GitHub 새 레포 업로드 가이드

## 전제 조건

- GitHub 계정에 로그인된 상태
- `gh` CLI 또는 `git` + GitHub Personal Access Token(PAT) 준비

---

## 방법 1: `gh` CLI 사용 (권장)

### 1단계 — gh CLI 로그인 확인

```bash
gh auth status
```

로그인이 안 된 경우:

```bash
gh auth login
```

### 2단계 — git 초기화 및 첫 커밋

```bash
cd /sjw_alinlab1/home/jungwook/action_tokenizer

git init
git add .
git commit -m "Initial commit"
```

> **주의:** `rla-wm`처럼 내부에 `.git`이 있는 디렉토리가 있으면 embedded git repository 경고가 뜨며, 실제 파일이 아닌 gitlink 포인터로만 커밋됩니다. 아래 2-1단계로 해결하세요.

### 2-1단계 — embedded git 레포 (rla-wm) 처리

내부 코드를 직접 포함시키려면:

```bash
git rm --cached rla-wm
rm -rf rla-wm/.git          # rla-wm의 git 이력만 제거 (코드 파일은 유지됨)
git add rla-wm/
git commit -m "Include rla-wm source files directly"
```

### 3단계 — GitHub에 새 레포 생성 및 push

```bash
# 공개 레포로 생성
gh repo create action-tokenizer --public --source=. --remote=origin --push

# 비공개 레포로 생성하려면 --private 사용
# gh repo create action-tokenizer --private --source=. --remote=origin --push
```

> `action-tokenizer` 부분은 원하는 레포 이름으로 변경하세요.

---

## 방법 2: `git` + PAT 사용 (gh CLI 없는 경우)

### 1단계 — GitHub에서 레포 생성

1. https://github.com/new 접속
2. Repository name 입력 (예: `action-tokenizer`)
3. Public / Private 선택
4. **"Add a README file" 체크 해제** (이미 README.md 있음)
5. Create repository 클릭

### 2단계 — git 초기화 및 첫 커밋

```bash
cd /sjw_alinlab1/home/jungwook/action_tokenizer

git init
git add .
git commit -m "Initial commit"
```

### 3단계 — 원격 레포 연결 및 push

```bash
# <YOUR_USERNAME>과 <REPO_NAME>을 실제 값으로 교체
git remote add origin https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
git branch -M main
git push -u origin main
```

PAT 인증이 필요한 경우 username 입력 후 password 자리에 PAT를 붙여넣으세요.

---

## 참고: `.claude/` 디렉토리 제외 여부 확인

현재 `.gitignore`에 `.claude/`가 포함되어 있지 않습니다.  
Claude Code 설정 파일을 레포에 포함시키지 않으려면 아래 명령으로 추가하세요:

```bash
echo ".claude/" >> .gitignore
git add .gitignore
git commit -m "Exclude .claude directory from version control"
```
