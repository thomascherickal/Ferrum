OWNER="thomascherickal"
REPOS=("ferrum-cpu" "ferrum-gpu" "ferrum-ml" "ferrum-junk")

# Untrack any stale gitlink or directory cache in the parent
for repo in "${REPOS[@]}"; do
  git rm --cached "$repo" 2>/dev/null || true
done

# Register each sub-directory as a submodule with SSH URL
for repo in "${REPOS[@]}"; do
  git submodule add "git@github.com:$OWNER/$repo.git" "$repo"
done

# Stage .gitmodules and submodule references, then commit
git add .gitmodules "${REPOS[@]}"
git commit -m "feat: link ferrum submodules over SSH"

# Push parent repository over SSH
git remote remove origin 2>/dev/null || true
git remote add origin "git@github.com:$OWNER/ferrum.git"

PARENT_BRANCH=$(git branch --show-current)
git push --recurse-submodules=on-demand -u origin "$PARENT_BRANCH" --force
