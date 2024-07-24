# When you add a new target variable, add it to the TYPE_COLS and CLASS_COLS lists
# Do not make any change in the model files
class Config:
    """
    Configuration class to store all the constants
    """
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COLS = ['y2', 'y3', 'y4'] # add new target variables here for example y5
    TARGET_COLS_NAME = {'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'} # Name mapping
    GROUPED = 'y1'
