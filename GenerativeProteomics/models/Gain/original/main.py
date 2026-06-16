from pathlib import Path

from GenerativeProteomics.evaluation.cross_validation_protogain import CrossValidatorProtoGain

def main() -> None:
    cv_protogain = CrossValidatorProtoGain(
        n_folds=5,
        seed=42,
        missing_file=Path("../../../data/raw/PXD016999.2.tsv"),
    )

    cv_protogain.run()

if __name__ == "__main__":
    main()