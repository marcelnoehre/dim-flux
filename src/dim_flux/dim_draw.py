import re
import subprocess
from pathlib import Path

from src.utils.variables import Variables


class DimDraw():
    '''
    Disclaimer
    ----------
    The original version is integrated into the tool conexp-clj [see https://github.com/tomhanika/conexp-clj].

    Reference
    ---------
    @misc{dürrschnabel2019dimdrawnoveltool,
        title={DimDraw -- A novel tool for drawing concept lattices},
        author={Dominik Dürrschnabel and Tom Hanika and Gerd Stumme},
        year={2019},
        eprint={1903.00686},
        archivePrefix={arXiv},
        primaryClass={cs.CG},
        url={https://arxiv.org/abs/1903.00686}
    }
    '''
    def __init__(self,
            variables: Variables
        ):
        '''
        Initialize DimDraw with a given 'realizer'.

        Parameters
        ----------
        variables: Variables
            The storage of variables
        '''
        self.vars = variables
        self.concepts = {
            c: self.vars.extents[c] | self.vars.intents[c]
            for c in self.vars.concepts
        }

    def two_dimensional_extension(self):
        '''
        Compute the two-dimensional extension of the lattice.

        Raises
        ------
        error : CalledProcessError
            If the script cannot find or run the provided JAR-file of brunt.

        Returns
        -------
        realizer : List
            Two-dimensional extension of the concept lattice.
        '''
        le_x, le_y = [None] * self.vars.N_c, [None] * self.vars.N_c

        try:
            # execute conexp-clj
            base_dir = Path(__file__).resolve().parents[2]
            jar_path = base_dir / "libs" / "brunt-fork.jar"

            if self.vars.cxt.endswith('.cxt'):
                cxt_path = Path(self.vars.cxt).resolve()
                res = subprocess.check_output(
                    ["java", "-jar", str(jar_path), "-f", "dim-draw-coordinates", str(cxt_path)],
                    text=True,
                    stderr=subprocess.STDOUT
                )
            else:
                cxt_path = Path(f'data/{self.vars.cxt}.cxt').resolve()
                res = subprocess.check_output(
                    ["java", "-jar", str(jar_path), "-f", "dim-draw-coordinates", str(cxt_path)],
                    text=True,
                    stderr=subprocess.STDOUT
                )
            for line in res.splitlines():
                concept, coords = line.split(' -> ')
                x, y = coords.strip("()").split(", ")
                node = next((k for k, v in self.concepts.items() if v == set(re.findall(r"\b[g|m][A-Za-z0-9]+\b", concept))), None)
                le_x[int(x)] = node
                le_y[int(y)] = node

            self.realizer = [le_x, le_y]
            return self.realizer
        except subprocess.CalledProcessError as e:
            raise ValueError(f'Error running JAR: {e}')