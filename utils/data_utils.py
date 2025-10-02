import pandas as pd
from sklearn.model_selection import train_test_split
import os


def load_and_split_cropped_dataset(cropped_dirs, metadata_csv):
    """
    Args:
        cropped_dirs (list): List of directories containing cropped images.
        metadata_csv (str): Path to metadata CSV.
    Returns:
        train_df, val_df, test_df: DataFrames split from matched metadata.
    """
    excluded_base_names = [
        "s-prd-462542531.jpg",
        "s-prd-567681349.jpg",
        "s-prd-595361939.jpg",
        "s-prd-719354460.jpg",
        "s-prd-752575241.jpg",
        "s-prd-470472240.jpg",
        "s-prd-601218250.jpg",
        "s-prd-632469223.jpg",
        "s-prd-653536778.jpg",
        "s-prd-767507626.jpg"
    ]
    df = pd.read_csv(metadata_csv)

    # Filter out control images
    df = df[df['midas_iscontrol'].str.lower() == 'no']

    # Normalize fields
    df['midas_distance'] = df['midas_distance'].str.lower()

    # Define modality
    df['modality'] = df['midas_distance'].apply(
        lambda x: 'dermoscope' if x == 'dscope' else ('clinical' if isinstance(x, str) else None)
    )

    # Filter to only 6in clinical images
    df = df[(df['modality'] == 'clinical') & (df['midas_distance'] == '6in')]

    # Assign label based on midas_path
    df['label'] = df['midas_path'].str.lower().str.contains("malig").astype(int)

    # Prepare for matching
    df['base_name'] = df['midas_file_name'].apply(lambda x: os.path.splitext(x)[0])
    metadata_lookup = df.set_index('base_name')

    metadata_lookup = metadata_lookup.drop(excluded_base_names, errors='ignore')

    # Match across all cropped directories
    matched_rows = []
    for cropped_dir in cropped_dirs:
        for f in os.listdir(cropped_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                base = os.path.splitext(f)[0]
                if base in metadata_lookup.index:
                    row = metadata_lookup.loc[base]
                    matched_rows.append({
                        'midas_record_id': row['midas_record_id'],
                        'clinical_path': os.path.join(cropped_dir, f),
                        'label': row['label'],
                        'clinical_midas_distance': row['midas_distance']
                    })

    matched_df = pd.DataFrame(matched_rows)

    # Split into train, val, test
    train_val, test_df = train_test_split(
        matched_df, test_size=0.15, random_state=42, stratify=matched_df["label"]
    )
    train_df, val_df = train_test_split(
        train_val, test_size=0.1765, random_state=42, stratify=train_val["label"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def generate_kfold_csv(
    cropped_dirs, 
    metadata_csv, 
    output_csv="kfold_midas.csv", 
    n_splits=5, 
    random_state=42, 
    test_size=0.15
):
    excluded_base_names = [
        "s-prd-462542531.jpg",
        "s-prd-567681349.jpg",
        "s-prd-595361939.jpg",
        "s-prd-719354460.jpg",
        "s-prd-752575241.jpg",
        "s-prd-470472240.jpg",
        "s-prd-601218250.jpg",
        "s-prd-632469223.jpg",
        "s-prd-653536778.jpg",
        "s-prd-767507626.jpg"
    ]

    df = pd.read_csv(metadata_csv)
    df = df[df['midas_iscontrol'].str.lower() == 'no']
    df['midas_distance'] = df['midas_distance'].str.lower()
    df['modality'] = df['midas_distance'].apply(
        lambda x: 'dermoscope' if x == 'dscope' else ('clinical' if isinstance(x, str) else None)
    )
    df = df[(df['modality'] == 'clinical') & (df['midas_distance'] == '6in')]
    df['label'] = df['midas_path'].str.lower().str.contains("malig").astype(int)
    df['base_name'] = df['midas_file_name'].apply(lambda x: os.path.splitext(x)[0])
    metadata_lookup = df.set_index('base_name')
    metadata_lookup = metadata_lookup.drop(excluded_base_names, errors='ignore')

    matched_rows = []
    for cropped_dir in cropped_dirs:
        for f in os.listdir(cropped_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                base = os.path.splitext(f)[0]
                if base in metadata_lookup.index:
                    row = metadata_lookup.loc[base]
                    matched_rows.append({
                        'midas_record_id': row['midas_record_id'],
                        'clinical_path': os.path.join(cropped_dir, f),
                        'label': row['label'],
                        'clinical_midas_distance': row['midas_distance'],
                        'base_name': base
                    })

    matched_df = pd.DataFrame(matched_rows).reset_index(drop=True)

    # Split out test set stratified by label
    train_val_df, test_df = train_test_split(
        matched_df, 
        test_size=test_size, 
        stratify=matched_df['label'], 
        random_state=random_state
    )

    # Assign K-Fold only on train_val_df
    train_val_df['fold'] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_val_df, train_val_df["label"])):
        train_val_df.iloc[val_idx, train_val_df.columns.get_loc("fold")] = fold_idx

    # Mark test set fold as -1 (or some other sentinel)
    test_df['fold'] = -1

    # Combine back train_val and test sets for full output CSV
    combined_df = pd.concat([train_val_df, test_df]).reset_index(drop=True)

    combined_df.to_csv(output_csv, index=False)
    print(f"Saved K-Fold + Test split CSV to {output_csv}")

    return combined_df, test_df



def load_train_and_val_df(image_directory, metadata_csv):
    """
    Load training and validation data from a CSV file and image directory.

    Args:
        image_directory: Path to the directory containing images.
        metadata_csv: Path to the CSV file containing metadata.
    Returns:
        train_df: DataFrame containing training data.
        val_df: DataFrame containing validation data.
    """
    df = pd.read_csv(metadata_csv)

    # Remove control examples
    df = df[df['midas_iscontrol'].str.lower() == 'no']

    # Normalize casing
    df['midas_distance'] = df['midas_distance'].str.lower()

    # Determine modality
    df['modality'] = df['midas_distance'].apply(
        lambda x: 'dermoscope' if x == 'dscope' else ('clinical' if isinstance(x, str) else None)
    )

    # Add existence flag
    df['image_path'] = df['midas_file_name'].apply(lambda x: os.path.join(image_directory, x))
    df['file_exists'] = df['image_path'].apply(os.path.exists)

    # Keep only valid images
    df = df[df['file_exists'] & df['modality'].isin(['dermoscope', 'clinical'])]

    # Only keep patients with at least 1 of each modality
    grouped = df.groupby('midas_record_id')
    valid_patients = [
        pid for pid, g in grouped
        if 'dermoscope' in g['modality'].values and 'clinical' in g['modality'].values
    ]

    # Filter down
    df = df[df['midas_record_id'].isin(valid_patients)]

    # For each patient, sample exactly 1 of each modality
    def sample_patient_rows(group):
        dermo = group[group['modality'] == 'dermoscope'].sample(1)
        clinical = group[group['modality'] == 'clinical'].sample(1)
        return pd.concat([dermo, clinical])

    final_df = df.groupby('midas_record_id').apply(sample_patient_rows).reset_index(drop=True)

    grouped = df.groupby('midas_record_id')
    valid_patients = [
        pid for pid, g in grouped
        if 'dermoscope' in g['modality'].values and 'clinical' in g['modality'].values
    ]

    # Assign labels: 1 = malignant, 0 = benign
    # Only use rows that have path info (non-control)
    final_df = final_df[final_df['midas_iscontrol'].str.lower() == 'no']
    final_df['label'] = final_df['midas_path'].str.lower().str.contains("malig").astype(int)

    # Restructure into per-patient rows
    patients = []
    for pid, g in final_df.groupby('midas_record_id'):
        if len(g) != 2: continue
        dermo_row = g[g['modality'] == 'dermoscope'].iloc[0]
        clin_row = g[g['modality'] == 'clinical'].iloc[0]
        label = dermo_row['label']  # same for both rows
        patients.append({
            'midas_record_id': pid,
            'dermo_path': dermo_row['image_path'],
            'clinical_path': clin_row['image_path'],
            'dermo_midas_distance': dermo_row['midas_distance'],
            'clinical_midas_distance': clin_row['midas_distance'],
            'label': label
        })

    patient_df = pd.DataFrame(patients)

    # Train/Val Split (patient-level)
    train_df, val_df = train_test_split(
        patient_df, test_size=0.2, random_state=42, stratify=patient_df['label']
    )

    return train_df, val_df


def load_train_val_test_df(image_directory, metadata_csv, val_size=0.1, test_size=0.2, random_state=42):
    """
    Load training, validation, and test data from a CSV file and image directory.

    Args:
        image_directory: Path to the directory containing images.
        metadata_csv: Path to the CSV file containing metadata.
        val_size: Proportion of train_val to use as validation.
        test_size: Proportion of total data to use as test set.
        random_state: Random seed for reproducibility.
    Returns:
        train_df, val_df, test_df: DataFrames for train, validation, and test sets.
    """
    df = pd.read_csv(metadata_csv)

    # Remove control examples
    df = df[df['midas_iscontrol'].str.lower() == 'no']

    # Normalize casing
    df['midas_distance'] = df['midas_distance'].str.lower()

    # Determine modality
    df['modality'] = df['midas_distance'].apply(
        lambda x: 'dermoscope' if x == 'dscope' else ('clinical' if isinstance(x, str) else None)
    )

    # Add image existence flag
    df['image_path'] = df['midas_file_name'].apply(lambda x: os.path.join(image_directory, x))
    df['file_exists'] = df['image_path'].apply(os.path.exists)

    # Keep valid rows only
    df = df[df['file_exists'] & df['modality'].isin(['dermoscope', 'clinical'])]

    # Only keep patients with both dermoscope and clinical images
    grouped = df.groupby('midas_record_id')
    valid_patients = [
        pid for pid, g in grouped
        if 'dermoscope' in g['modality'].values and 'clinical' in g['modality'].values
    ]
    df = df[df['midas_record_id'].isin(valid_patients)]

    # Sample exactly 1 of each modality per patient
    def sample_patient_rows(group):
        dermo = group[group['modality'] == 'dermoscope'].sample(1, random_state=random_state)
        clinical = group[group['modality'] == 'clinical'].sample(1, random_state=random_state)
        return pd.concat([dermo, clinical])

    final_df = df.groupby('midas_record_id').apply(sample_patient_rows).reset_index(drop=True)

    # Assign binary labels
    final_df['label'] = final_df['midas_path'].str.lower().str.contains("malig").astype(int)

    # Create per-patient rows
    patients = []
    for pid, g in final_df.groupby('midas_record_id'):
        if len(g) != 2: continue
        dermo_row = g[g['modality'] == 'dermoscope'].iloc[0]
        clin_row = g[g['modality'] == 'clinical'].iloc[0]
        label = dermo_row['label']
        patients.append({
            'midas_record_id': pid,
            'dermo_path': dermo_row['image_path'],
            'clinical_path': clin_row['image_path'],
            'dermo_midas_distance': dermo_row['midas_distance'],
            'clinical_midas_distance': clin_row['midas_distance'],
            'label': label
        })

    patient_df = pd.DataFrame(patients)

    # First split into train_val and test
    train_val_df, test_df = train_test_split(
        patient_df, test_size=test_size, random_state=random_state, stratify=patient_df['label']
    )

    # Then split train_val into train and val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df['label']
    )

    return train_df, val_df, test_df

