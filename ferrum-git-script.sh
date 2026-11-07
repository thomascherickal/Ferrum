cd /path/to/ferrum

OWNER="thomascherickal"
REPOS=("ferrum-cpu" "ferrum-gpu" "ferrum-ml" "ferrum-junk")

for repo in "${REPOS[@]}"; do
  echo "===> Setting up remote and pushing: $repo..."

  # 1. Create the remote repository on GitHub via gh (change --public to --private if desired)
  gh repo create "$OWNER/$repo" --public --confirm=false 2>/dev/null || true

  # 2. Enter sub-repo directory
  cd "$repo"

  # 3. Ensure Git is initialized and contains an initial commit
  if [ ! -d ".git" ]; then
    git init -b main
    git add .
    git commit -m "chore: initial commit for $repo"
  fi

  # 4. Attach SSH remote origin
  git remote remove origin 2>/dev/null || true
  git remote add origin "git@github.com:$OWNER/$repo.git"

  # 5. Push current branch using SSH
  BRANCH=$(git branch --show-current)
  [ -z "$BRANCH" ] && BRANCH="main"
  git push -u origin "$BRANCH" --force

  # Return to ferrum root
  cd ..
done