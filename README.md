# pds_project_skillpro

SkillMatch Pro: Intelligent Resume Insights 
Description
SkillMatch Pro: Intelligent Resume Insights is a web application that helps job seekers understand how well their resume matches job descriptions in their desired industry. The application uses natural language processing (NLP) and machine learning algorithms to analyze and identify the most relevant skills for a given job category. The application provides insights on the predicted job category, top skills for that category, skills present in the resume, missing skills, and the chances of getting shortlisted based on the resume content.
Technologies Used
•	Python
•	Flask
•	Pandas
•	NumPy
•	Scikit-Learn
•	PyPDF2
•	Regex
•	Seaborn
•	Matplotlib
•	HTML, CSS, and JavaScript (for the frontend)
Project Structure
•	app.py: The main application file that initializes and runs the Flask web server.
•	data_preprocessing.py: Contains functions for cleaning and preprocessing resume text.
•	data_transformation.py: Handles the loading and processing of the skills dataset.
•	modeling.py: Creates and trains the machine learning pipeline used for predictions.
•	results.py: Generates insights and results from the model's predictions.
•	index.html: The main HTML template for the web application.
How to Run
1.	Ensure you have Python 3 installed.
2.	Install the required Python packages using the following command:
pip install Flask pandas numpy scikit-learn PyPDF2 seaborn matplotlib 
3.	Download the necessary data files (resume_dataset.csv and skills_30k.csv) and place them in the appropriate directory.
4.	Run the application using the following command:
python app.py 
5.	Open your web browser and navigate to http://127.0.0.1:5000/ to view the application.
Usage
1.	Upload your resume in PDF format by clicking on the "Choose File" button.
2.	Click on the "Analyze Resume" button to start the analysis.
3.	The application will display the predicted job category, top skills for that category, skills present in your resume, missing skills, and your chances of getting shortlisted.
4.	Optionally, you can view the top skills for different job categories by clicking on the respective category buttons.

![image](https://user-images.githubusercontent.com/98427744/236995901-587362bd-7b35-4505-8e7d-8dfae29610bc.png)
