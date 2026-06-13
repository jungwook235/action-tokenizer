# GitHub 업로드 가이드

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
