# Project Instructions

In this project, your task is to build a system for predicting median
household income using data aggregated from a number of questionaires
filled out by volunteers. Your system will output a categorical label 
between 1 and 9 inclusive. Your goal is to use the provided `train_data.txt`
which is labelled to build a model that you will then use to get predictions
for the unlabelled `test_data.txt`. Your model's predictions will be
evaluated according to accuracy. 

# Data Format

There are 8093 training samples provided. The data is provided with each
sample on a comma-delimited row, where each column corresponds to a different
feature. These features are as follows:

- Column 1: Income
    - This is the categorical label being predicted.
    - The values correspond to the following mapping:
         - (1) Less than $10,000
         - (2) $10,000 to $14,999
         - (3) $15,000 to $19,999
         - (4) $20,000 to $24,999
         - (5) $25,000 to $29,999
         - (6) $30,000 to $39,999
         - (7) $40,000 to $49,999
         - (8) $50,000 to $74,999
         - (9) $75,000 or more

         
- Column 2: Marital Status
    - The values correspond to the following mapping:
         - (1) Married
         - (2) Living together, not married
         - (3) Divorced or separated
         - (4) Widowed
         - (5) Single, never married
         
- Column 3: Age
    - The values correspond to the following mapping:
         - (1) 14 thru 17
         - (2) 18 thru 24
         - (3) 25 thru 34
         - (4) 35 thru 44
         - (5) 45 thru 54
         - (6) 55 thru 64
         - (7) 65 and Over
         
- Column 4: Education
    - The values correspond to the following mapping:
         - (1) Grade 8 or less 
         - (2) Grades 9 to 11
         - (3) Graduated high school
         - (4) 1 to 3 years of college
         - (5) College graduate
         - (6) Grad study
         
- Column 5: Occupation
    - The values correspond to the following mapping:
         - (1) Professional/Managerial
         - (2) Sales Worker
         - (3) Factory Worker/Laborer/Driver
         - (4) Clerical/Service Worker
         - (5) Homemaker
         - (6) Student, HS or College
         - (7) Military
         - (8) Retired
         - (9) Unemployed

- Column 6: How long a person has lived in the given area
    - The values correspond to the following mapping:
         - (1) Less than one year
         - (2) One to three years
         - (3) Four to six years
         - (4) Seven to ten years
         - (5) More than ten years

- Column 7: Dual Incomes (if married)
    - The values correspond to the following mapping:
         - (1) Not Married
         - (2) Yes
         - (3) No

- Column 8: Number Persons in Household
    - The values correspond to the following mapping:
         - (1) One
         - (2) Two
         - (3) 3
         - (4) 4
         - (5) 5
         - (6) 6
         - (7) 7
         - (8) 8
         - (9) 9 or more
         
- Column 9: Persons in Household Under 18
    - The values correspond to the following mapping:
         - (0) None
         - (1) One
         - (2) Two
         - (3) 3
         - (4) 4
         - (5) 5
         - (6) 6
         - (7) 7
         - (8) 8
         - (9) 9 or more 
         
- Column 10: Householder Status
    - The values correspond to the following mapping:
         - (1) Own
         - (2) Rent
         - (3) Live with parents/family         
 
 - Column 11: Type of Home
    - The values correspond to the following mapping:
         - (1) House
         - (2) Condominium
         - (3) Apartment
         - (4) Mobile Home
         - (5) Other       
         
        
         
NOTE: The value for some of the columns may be NA, indicating that this 
feature value is missing.


There are also 900 testing datapoints provided in `test_data.txt`. 
Each datapoint has the same columns as described above, but without the 
income label included. Your job is to provide labels for them.

# Output Format

Your goal is to generate a numerical label for the 900 testing datapoints.
You will output your labels as a newline-delimited text file with one
label per line. The label on the first line of your text file should
correspond to the datapoint in first row of the `test_data.txt` file, the
second line in the text file to the second row, etc.

For example, if you predicted five labels for the first datapoints: (1, 1, 4, 5, 3)
you would generate the following output text file:

```
1
1
4
5
3
```
