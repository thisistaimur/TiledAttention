# Releases

This repository uses Git tags and GitHub Actions to automate releases.

## Workflow Summary

1. Push a tag that matches `v*` (for example `v0.0.1`).
2. `.github/workflows/create-release-on-tag.yml` auto-creates and publishes a GitHub Release.
3. The tag workflow dispatches `.github/workflows/release-python-artifacts.yml` for that tag.
4. That workflow builds Python artifacts (`.whl`, `.tar.gz`) and attaches them to the same release.

## Tag Rules

- Stable release: `vX.Y.Z` (example: `v1.4.0`)
- Pre-release: include a suffix like `-rc1` (example: `v1.5.0-rc1`)
  - These are marked as `prerelease: true` automatically by workflow logic.

## Maintainer Steps

1. Update `project.version` in `pyproject.toml`.
2. Commit and push your changes to `main`.
3. Create and push a version tag:

```bash
git tag v0.0.1
git push origin v0.0.1
```

4. Wait for both GitHub Actions workflows to finish.
5. Verify release artifacts exist:
   - `tiledattention-<version>-py3-none-any.whl`
   - `tiledattention-<version>.tar.gz`

## Notes

- If you manually publish a release in the GitHub UI, the artifact workflow still runs because it listens to `release.published`.
- You can also run artifact upload manually:

```bash
gh workflow run release-python-artifacts.yml -f tag=v0.0.1
```

- This project distributes Python artifacts through GitHub Releases (not PyPI).
