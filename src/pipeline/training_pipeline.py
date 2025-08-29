"""
    Script to run the training pipeline for the model.
    :param file_path: Path to the dataset CSV file.
    :param target_column: Name of the target column.
    """
# from src.models.lr import main
from src.models.rf import main 
processed_columns = [
    'Item_Weight',
    'Item_Visibility',
    'Item_MRP',
    'Outlet_Establishment_Year',
    'Item_Identifier_encoded',
 'Item_Type_encoded',
 'Item_Fat_Content_encoded_LF',
 'Item_Fat_Content_encoded_REG',
 'Outlet_Location_Type_Tier 1',
 'Outlet_Location_Type_Tier 2',
 'Outlet_Location_Type_Tier 3',
 'Outlet_Size_High',
 'Outlet_Size_Medium',
 'Outlet_Size_Small',
 'Outlet_Type_Grocery Store',
 'Outlet_Type_Supermarket Type1',
 'Outlet_Type_Supermarket Type2',
 'Outlet_Type_Supermarket Type3',
 'Sales'
]

if __name__ == "__main__":
    # Example usage
    file_path = "data/processed/train.csv"  # Replace with your dataset path
    
    target_column = "Sales"  # Replace with your target column name
    main(file_path, target_column, processed_columns)