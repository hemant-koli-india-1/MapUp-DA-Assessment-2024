import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    required_columns = ['id_start', 'id_end', 'distance']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    all_ids = sorted(set(df['id_start']).union(df['id_end']))
    distance_matrix = pd.DataFrame(np.inf, index=all_ids, columns=all_ids)
    np.fill_diagonal(distance_matrix.values, 0)
    
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[id_start, id_end] = distance
        distance_matrix.at[id_end, id_start] = distance

    for k in all_ids:
        for i in all_ids:
            for j in all_ids:
                distance_matrix.at[i, j] = min(distance_matrix.at[i, j], 
                                               distance_matrix.at[i, k] + distance_matrix.at[k, j])
    
    return distance_matrix

def unroll_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(distance_matrix, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    data = []
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end and distance_matrix.at[id_start, id_end] < np.inf:
                distance = distance_matrix.at[id_start, id_end]
                data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    unrolled_df = pd.DataFrame(data)
    return unrolled_df

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if 'id_start' not in df.columns or 'distance' not in df.columns:
        raise ValueError("Missing required columns: 'id_start' or 'distance'.")

    reference_distances = df[df['id_start'] == reference_id]

    if reference_distances.empty:
        return pd.DataFrame(columns=['id_start'])

    average_distance = reference_distances['distance'].mean()
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1

    filtered_ids = df[
        (df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)
    ]

    result_ids = sorted(filtered_ids['id_start'].unique())
    return pd.DataFrame(result_ids, columns=['id_start'])

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if 'distance' not in df.columns:
        raise ValueError("Missing required column: 'distance'.")

    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate

    return df

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    if 'id_start' not in df.columns or 'id_end' not in df.columns:
        raise ValueError("Missing required columns: 'id_start' or 'id_end'.")

    df['id_start'] = df['id_start'].astype(int)
    df['id_end'] = df['id_end'].astype(int)

    weekday_discount_factors = {
        'morning': 0.8,
        'day': 1.2,
        'evening': 0.8
    }
    weekend_discount_factor = 0.7

    morning_start = pd.Timestamp("00:00:00").time()
    morning_end = pd.Timestamp("10:00:00").time()
    day_start = pd.Timestamp("10:00:00").time()
    day_end = pd.Timestamp("18:00:00").time()
    evening_start = pd.Timestamp("18:00:00").time()
    evening_end = pd.Timestamp("23:59:59").time()

    new_rows = []

    for (id_start, id_end) in df[['id_start', 'id_end']].drop_duplicates().values:
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            new_rows.append({'start_day': day, 'id_start': id_start, 'id_end': id_end, 'start_time': morning_start, 'end_time': morning_end})
            new_rows.append({'start_day': day, 'id_start': id_start, 'id_end': id_end, 'start_time': day_start, 'end_time': day_end})
            new_rows.append({'start_day': day, 'id_start': id_start, 'id_end': id_end, 'start_time': evening_start, 'end_time': evening_end})

    new_rows_df = pd.DataFrame(new_rows)

    merged_df = pd.merge(df, new_rows_df, on=['id_start', 'id_end'], how='outer')

    for index, row in merged_df.iterrows():
        start_day = row['start_day']
        start_time = row['start_time']

        if start_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            if morning_start <= start_time < morning_end:
                for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                    merged_df.at[index, vehicle_type] *= weekday_discount_factors['morning']
            elif day_start <= start_time < day_end:
                for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                    merged_df.at[index, vehicle_type] *= weekday_discount_factors['day']
            elif evening_start <= start_time <= evening_end:
                for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                    merged_df.at[index, vehicle_type] *= weekday_discount_factors['evening']
        else:
            for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                merged_df.at[index, vehicle_type] *= weekend_discount_factor

    return merged_df
