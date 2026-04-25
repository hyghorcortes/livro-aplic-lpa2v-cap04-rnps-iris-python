from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    ROOT / 'NN' / 'main_iris_nn_didatico.py',
    ROOT / 'CNAPa' / 'main_iris_cnapa_didatico.py',
    ROOT / 'CNAPap' / 'main_iris_cnapap_didatico_v2_autocontido.py',
    ROOT / 'CNAPCa' / 'main_iris_cnapca_didatico_autocontido.py',
]


def main() -> None:
    for script in SCRIPTS:
        print('\n' + '=' * 80)
        print(f'Executando: {script.relative_to(ROOT)}')
        print('=' * 80)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

    print('\nTodos os exemplos foram executados com sucesso.')


if __name__ == '__main__':
    main()
