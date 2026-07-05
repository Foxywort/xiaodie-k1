# GitHub Upload Notes

This local package is a full delivery bundle and is much larger than a normal GitHub source repository.

The GitHub repository intentionally tracks source code, configuration, documentation, reports, small demos, and hashes. It excludes large generated assets such as GGUF files, merged Hugging Face models, ONNX runtime models, training checkpoints, downloaded datasets, and board runtime copies.

Why these files are excluded:

- GitHub blocks normal Git pushes containing files over 100 MB.
- This package is about 37 GB locally, which is not practical for a normal GitHub repository.
- Model weights and training outputs are better stored with GitHub Releases, Git LFS with a paid quota, Hugging Face, ModelScope, cloud storage, or an internal artifact store.

Important local artifact references are preserved in `README_PACKAGE.md` and `IMPORTANT_HASHES.sha256`.
