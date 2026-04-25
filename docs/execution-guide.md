# Guia de execução

## 1. Criar ambiente virtual

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Instalar dependências

```bash
pip install -r requirements.txt
```

## 3. Executar todos os exemplos

```bash
python run_all_examples.py
```

## 4. Executar variantes individuais

```bash
python NN/main_iris_nn_didatico.py
python CNAPa/main_iris_cnapa_didatico.py
python CNAPap/main_iris_cnapap_didatico_v2_autocontido.py
python CNAPCa/main_iris_cnapca_didatico_autocontido.py
```

## 5. Conferir resultados

Após a execução, verifique os arquivos `.png`, `.csv`, `.json` e `.joblib` dentro da pasta de cada variante.
