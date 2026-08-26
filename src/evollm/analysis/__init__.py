"""Population analysis for evoLLM runs.

Typical use:

    from evollm.analysis import analyse_run, format_report
    print(format_report(analyse_run("runs/my_run")))

or piecewise, when you want to ask something the suite does not:

    from evollm.analysis import Pedigree, build_phenotypes, variance_partition
    ped   = Pedigree.from_run(run)
    pheno = build_phenotypes(run, pedigree=ped)
    variance_partition(pheno["move_share"], pheno["lineage"], n_perm=1000)
"""

from .genotypes import (align, build_genotypes, genotype_matrix,
                        load_fingerprints, load_genome_features,
                        snapshot_paths)
from .pedigree import Pedigree
from .phenotypes import (ALL_TRAITS, TRAIT_GROUPS, build_phenotypes,
                         strategy_matrix)
from .descent import Descent
from .traces import (build_bundle, format_inspection, inspect,
                     sample_for_review)
from .lifecourse import (action_curve, format_lifecourse,
                        surprise_curve)
from .stats import (associate, associate_alleles, benjamini_hochberg,
                    kmeans, mutual_information, ols,
                    parent_offspring_concordance, principal_components,
                    replication, residualise, sibling_concordance,
                    variance_partition)
from .suite import analyse_run, format_report, generation_band, strata_of
from .table import Table

__all__ = [
    "Table", "Pedigree",
    "build_phenotypes", "strategy_matrix", "TRAIT_GROUPS", "ALL_TRAITS",
    "build_genotypes", "genotype_matrix", "load_genome_features",
    "snapshot_paths", "load_fingerprints", "align",
    "Descent", "surprise_curve", "action_curve",
    "format_lifecourse", "inspect", "format_inspection",
    "sample_for_review", "build_bundle", "variance_partition", "associate", "associate_alleles",
    "replication", "kmeans", "mutual_information",
    "parent_offspring_concordance", "sibling_concordance",
    "principal_components", "residualise", "ols", "benjamini_hochberg",
    "analyse_run", "format_report", "generation_band", "strata_of",
]
