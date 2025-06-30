# TODO:
- finish img embedding script tomorrow


# Ben's UV Environment Setup (.ben/pytorch_uv)

This guide walks **new contributors** through creating their own Python 3.10 environment inside the hidden `.ben/` folder and making it available in Jupyter.

---

## 1. Prerequisites

* **uv** installed  
  ```bash
  curl -Ls https://astral.sh/uv/install.sh | bash
  export PATH="$HOME/.cargo/bin:$PATH"
  ```

* **Python 3.10** available as `python3.10`

---

## 2. Create & activate the env

From the **project root**:

```bash
uv venv .ben/pytorch_uv --python=python3.10
source .ben/pytorch_uv/bin/activate
```

Your prompt should now show `(pytorch_uv)`.

---

## 3. Install project dependencies

`.ben/pyproject.toml` already lists all required packages.

```bash
cd .ben
uv pip compile --output-file uv.lock pyproject.toml
uv pip sync uv.lock
cd ..
```

---

## 4. Register the Jupyter kernel (one‑off)

```bash
python -m ipykernel install --user   \
  --name=pytorch_uv   \
  --display-name "Python (.ben/pytorch_uv)"
  ```

Select **“Python (.ben/pytorch_uv)”** in VS Code or JupyterLab.

---

## 5. Day‑to‑day usage

```bash
# Activate
source .ben/pytorch_uv/bin/activate

# Deactivate when done
deactivate
```

## 6. To add new 
Add the dependancy to the toml file.

```bash
cd .ben
uv pip compile --output-file uv.lock pyproject.toml
uv pip sync uv.lock
```

---

### Git Tips

Add (or keep) these lines in `.gitignore` to avoid committing your compiled env:

```
.ben/pytorch_uv/
.ben/uv.lock
```

The `pyproject.toml` inside `.ben/` **is committed** so everyone sees your dependency list.

Happy coding! 🚀