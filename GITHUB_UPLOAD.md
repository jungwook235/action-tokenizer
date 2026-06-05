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
