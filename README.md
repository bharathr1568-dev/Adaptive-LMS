Adaptive-LMS — AI-Powered Adaptive Learning Management System
An AI-powered Learning Management System (LMS) designed to improve the learning experience through student performance analytics, teacher insights, parent monitoring, and AI-assisted learning analysis.
📌 Project Overview
Adaptive-LMS is a web-based learning management platform built to support students, teachers, and parents through a centralized interface. It combines learning management features with analytics and AI-based capabilities to help users understand student progress and learning behaviour.

The platform provides separate dashboards for different user roles, enabling students to access learning activities, teachers to manage learning content and assessments, and parents to monitor student progress.

The project also explores AI-assisted learning insights and gaze-based analysis to support a more adaptive and personalized learning experience.
🎯 Objectives
- Develop a centralized platform for managing learning activities.
- Provide role-based dashboards for students, teachers, and parents.
- Support course management, quizzes, and assessments.
- Analyze student performance through learning analytics.
- Explore AI-assisted insights for understanding learning progress.
- Integrate gaze detection and video-based analysis to investigate student attention and learning behaviour.
- Build a foundation for a more personalized and adaptive learning environment.
✨ Key Features
👨‍🎓 Student Dashboard
- Access the student learning interface.
- View available courses and learning activities.
- Participate in quizzes and assessments.
- Review learning progress and performance analytics.
👩‍🏫 Teacher Dashboard
- Manage learning activities and courses.
- Create questions and assessments.
- Review student performance.
- Access analytics to understand student progress.
👨‍👩‍👧 Parent Dashboard
- Access student-related progress information.
- Review available performance analytics.
- Support monitoring of learning progress.
🤖 AI-Assisted Learning Analysis
- Integrates AI-related functionality to support learning insights.
- Provides a foundation for AI-assisted interpretation of learning information.
- Supports the development of more personalized learning experiences.
👁️ Gaze Detection and Video Analysis
- Includes gaze detection and video-analysis modules.
- Explores visual behaviour analysis as an additional source of learning-related information.
- Provides a foundation for investigating attention-aware learning systems.
Note: The exact availability of individual features depends on their current implementation and configuration in the repository.
🛠️ Technology Stack
Technology	Purpose
Python	Backend development
Flask	Web application framework
HTML	Web page structure
CSS	Styling and layout
JavaScript	Frontend interactions
SQLite / PostgreSQL	Database options, depending on the active configuration
Groq API	AI integration, where configured
OpenCV and related libraries	Computer vision and video analysis, where used
Gunicorn	Production application server
Render	Cloud deployment
📂 Project Structure
Adaptive-LMS/
│
├── app.py                 # Main Flask application
├── video_ai.py            # AI/video-analysis functionality
├── gaze_detector.py       # Gaze detection module
│
├── gaze_module/           # Gaze-related components
│
├── static/                # CSS, JavaScript, images, and other assets
│
├── templates/             # HTML templates and dashboard pages
│
├── requirements.txt       # Python dependencies
├── .gitignore             # Files excluded from Git tracking
└── README.md              # Project documentation


⚙️ Installation and Setup
Follow these steps to run the project locally.
Prerequisites
Make sure you have installed:
- Python 3.10 or a compatible version supported by the project dependencies.
- Git.
- A code editor such as Visual Studio Code or Antigravity IDE.
- API credentials if AI-related features require them.
Step 1: Clone the repository
git clone https://github.com/bharathr1568-dev/Adaptive-LMS.git


Step 2: Navigate to the project directory
cd Adaptive-LMS


Step 3: Create a virtual environment
python -m venv .venv


Step 4: Activate the virtual environment
Windows:
.venv\Scripts\activate


Linux / macOS:
source .venv/bin/activate


Step 5: Install dependencies
pip install -r requirements.txt


Step 6: Configure environment variables
Add the API credentials and database configuration required by your current application.

For example:
GROQ_API_KEY=your_groq_api_key
GROK_API_KEY=your_grok_api_key
DATABASE_URL=your_database_connection_url


Only configure the variables that your code actually uses. GROQ_API_KEY and GROK_API_KEY refer to different services, so check the relevant code before setting them.

For local development, use a .env file only if your application is configured to load it. Never upload .env files or API credentials to GitHub.

Database note: Setting DATABASE_URL alone does not migrate an application from SQLite to PostgreSQL. The application code must support PostgreSQL and use the configured connection URL.
Step 7: Run the application
python app.py


If the application has the appropriate Flask startup configuration, the terminal will display the local development address. Open that address in your browser.

If the application does not start, check the error message, installed dependencies, environment variables, and database configuration.
🌐 Deployment
The project has been associated with a Render deployment.

Deployment URL: https://adaptive-lms-xsxl.onrender.com

The following configuration is commonly used when deploying this Flask application to Render.

Build Command
pip install -r requirements.txt


Start Command
gunicorn app:app


Deployment Steps
1. Push the latest project changes to GitHub.
2. Create or open the corresponding Web Service in Render.
3. Connect the GitHub repository.
4. Configure the build and start commands.
5. Add the required environment variables in Render's environment settings.
6. Configure the database connection if PostgreSQL support has been implemented in the application.
7. Deploy and check the application logs for errors.
Important: The application code, database schema, and environment configuration must all agree. If the code still uses SQLite, adding a PostgreSQL URL in Render will not automatically switch the database.
📊 Potential Applications
- Personalized online learning platforms.
- Student performance monitoring.
- Digital course and assessment management.
- Learning analytics and progress reporting.
- Research into attention-aware and adaptive learning systems.
- AI-assisted educational analysis.
🔮 Future Enhancements
- Complete and strengthen authentication and role-based access control.
- Integrate PostgreSQL for reliable multi-user data management.
- Improve student performance analytics and visualization.
- Develop more personalized learning recommendations.
- Enhance gaze detection and video-analysis capabilities.
- Improve error handling, database reliability, and application security.
- Add automated testing and comprehensive API documentation.
- Improve deployment monitoring and production readiness.
🔐 Security Considerations
- Store API keys and credentials in environment variables.
- Never commit passwords, secret keys, or database credentials.
- Use secure password hashing and appropriate session management.
- Validate user input and apply authorization checks to protected routes.
- Use a persistent database for production data.
- Avoid exposing sensitive student information through public endpoints or logs.
👨‍💻 Author
Bharath R.

GitHub: bharathr1568-dev

