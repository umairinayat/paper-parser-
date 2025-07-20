# Research Paper Analyzer

A comprehensive Django-based web application for searching, analyzing, and managing research papers using AI-powered tools.

## Features

- **AI-Powered Paper Search**: Search across multiple academic sources (ArXiv, Google Scholar, Nature, PubMed, IEEE, etc.)
- **Comprehensive Analysis**: AI-driven analysis of research papers including methodology, findings, and impact assessment
- **Paper Management**: Upload, organize, and track your research papers
- **Export Capabilities**: Export analysis results in CSV format
- **User Authentication**: Secure user accounts and paper management
- **Real-time Search**: Get up-to-date research papers from various academic sources

## Technology Stack

- **Backend**: Django 4.x
- **Database**: SQLite (development), PostgreSQL (production)
- **AI/ML**: LangChain, Groq API
- **Search**: ArXiv API, DuckDuckGo, Google Custom Search API
- **Frontend**: Bootstrap 5, jQuery
- **Python**: 3.12+

## Installation

### Prerequisites

- Python 3.12 or higher
- pip
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/research-paper-analyzer.git
   cd research-paper-analyzer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   # On Windows
   myenv\Scripts\activate
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your-secret-key-here
   DEBUG=True
   GROQ_API_KEY=your-groq-api-key
   GOOGLE_API_KEY=your-google-api-key
   GOOGLE_CSE_ID=your-google-cse-id
   ```

5. **Run migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

7. **Run the development server**
   ```bash
   python manage.py runserver
   ```

8. **Access the application**
   Open your browser and go to `http://127.0.0.1:8000`

## API Keys Setup

### Groq API
1. Sign up at [Groq Console](https://console.groq.com/)
2. Get your API key
3. Add to `.env` file: `GROQ_API_KEY=your-key`

### Google Custom Search API (Optional)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Custom Search API
3. Create API key
4. Set up Custom Search Engine at [Google Programmable Search Engine](https://programmablesearchengine.google.com/)
5. Add to `.env` file:
   ```env
   GOOGLE_API_KEY=your-api-key
   GOOGLE_CSE_ID=your-search-engine-id
   ```

## Usage

### Searching Papers
1. Navigate to "Search Papers" in the navigation
2. Enter your research query in natural language
3. Select the number of papers to analyze
4. Click "Search & Analyze Papers"
5. View comprehensive analysis results

### Uploading Papers
1. Go to "Upload Paper" in the navigation
2. Fill in paper details and upload PDF
3. The system will process and analyze your paper

### Managing Papers
1. View all your papers in "My Papers"
2. Edit paper metadata
3. View detailed analysis results
4. Export analysis data

## Project Structure

```
research-paper-analyzer/
├── analysis/                 # Analysis app
├── papers/                   # Papers app
│   ├── services/            # AI search agents
│   ├── models.py           # Paper models
│   ├── views.py            # Paper views
│   └── forms.py            # Paper forms
├── users/                   # User management
├── templates/               # HTML templates
├── static/                  # CSS, JS, images
├── media/                   # Uploaded files
├── logs/                    # Application logs
└── manage.py               # Django management
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Django](https://www.djangoproject.com/) - Web framework
- [LangChain](https://langchain.com/) - AI/ML framework
- [Groq](https://groq.com/) - AI inference platform
- [Bootstrap](https://getbootstrap.com/) - CSS framework

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

## Roadmap

- [ ] Add more academic sources
- [ ] Implement paper recommendation system
- [ ] Add citation analysis
- [ ] Support for more export formats
- [ ] Mobile app version
- [ ] Collaborative features
