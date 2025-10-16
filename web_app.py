"""
Web interface for the Advanced Document Search Engine
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from typing import Dict, Any

from .integration import initialize_search_engine, SearchAPI
from .config import search_config
from .compliance_dashboard import compliance_dashboard_bp
from .compliance_integration import compliance_integration_manager

# Initialize search components
search_engine, integrator, api = initialize_search_engine()

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Register compliance dashboard blueprint
app.register_blueprint(compliance_dashboard_bp)


@app.route('/')
def index():
    """Main search page"""
    return render_template('search.html')


@app.route('/api/search', methods=['GET'])
def api_search():
    """Search API endpoint"""
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 20))
    category = request.args.get('category', '')
    doc_type = request.args.get('type', '')

    # Build filters
    filters = {}
    if category:
        filters['category'] = category
    if doc_type:
        filters['document_type'] = doc_type

    # Perform search
    result = api.search(query, limit=limit, filters=filters)

    if result['success']:
        # Enhance results with compliance information
        enhanced_results = compliance_integration_manager.enhance_search_results(result['data']['results'])

        # Update the results in the response data
        result['data']['results'] = enhanced_results

        # Add compliance facets
        compliance_facets = compliance_integration_manager.add_compliance_search_facets(enhanced_results)
        result['data']['facets']['compliance'] = compliance_facets

        return jsonify(result['data'])
    else:
        return jsonify({'error': result['error']}), 500


@app.route('/api/suggestions', methods=['GET'])
def api_suggestions():
    """Search suggestions API endpoint"""
    query = request.args.get('q', '')
    result = api.get_suggestions(query)

    if result['success']:
        return jsonify(result['data'])
    else:
        return jsonify({'error': result['error']}), 500


@app.route('/api/document/<doc_id>', methods=['GET'])
def api_document(doc_id):
    """Get document details API endpoint"""
    result = api.get_document(doc_id)

    if result['success']:
        return jsonify(result['data'])
    else:
        return jsonify({'error': result['error']}), 404


@app.route('/api/popular', methods=['GET'])
def api_popular():
    """Get popular documents API endpoint"""
    limit = int(request.args.get('limit', 10))
    result = api.get_popular_documents(limit)

    if result['success']:
        return jsonify(result['data'])
    else:
        return jsonify({'error': result['error']}), 500


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get search engine statistics API endpoint"""
    result = api.get_stats()

    if result['success']:
        return jsonify(result['data'])
    else:
        return jsonify({'error': result['error']}), 500


@app.route('/api/track/<doc_id>', methods=['POST'])
def api_track(doc_id):
    """Track document access API endpoint"""
    result = api.track_document_access(doc_id)

    if result['success']:
        return jsonify(result['data'])
    else:
        return jsonify({'error': result['error']}), 500


@app.route('/search')
def search_page():
    """Advanced search page"""
    return render_template('advanced_search.html')


@app.route('/document/<doc_id>')
def document_page(doc_id):
    """Document view page"""
    return render_template('document.html', doc_id=doc_id)


@app.route('/browse')
def browse_page():
    """Browse documents page"""
    return render_template('browse.html')


@app.route('/compliance/dashboard')
def compliance_dashboard():
    """Compliance monitoring dashboard"""
    return render_template('compliance_dashboard.html')


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500


def create_templates():
    """Create HTML templates for the web interface"""
    templates_dir = Path('search_engine/templates')
    templates_dir.mkdir(exist_ok=True)

    # Main search template
    search_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Document Search</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .search-container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .search-box { margin-bottom: 30px; }
        .result-card { margin-bottom: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .result-title { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
        .result-meta { color: #666; font-size: 0.9em; margin-bottom: 15px; }
        .result-snippet { margin-bottom: 15px; }
        .highlight { background-color: #ffff00; padding: 2px 4px; border-radius: 3px; }
        .facets { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .facet-group { margin-bottom: 20px; }
        .facet-title { font-weight: bold; margin-bottom: 10px; }
        .stats { background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">📄 Advanced Document Search</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/">Search</a>
                <a class="nav-link" href="/browse">Browse</a>
                <a class="nav-link" href="/search">Advanced</a>
                <a class="nav-link" href="/compliance/dashboard">📋 Compliance</a>
            </div>
        </div>
    </nav>

    <div class="search-container">
        <div class="search-box">
            <form id="searchForm" class="d-flex">
                <input type="text" id="searchInput" class="form-control me-2" placeholder="Search documents... (e.g., 'invoice processing oracle cloud')" value="">
                <button type="submit" class="btn btn-primary">Search</button>
            </form>
            <div id="suggestions" class="mt-2" style="display: none;"></div>
        </div>

        <div id="loading" class="text-center" style="display: none;">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>

        <div id="results">
            <!-- Search results will be displayed here -->
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const searchInput = document.getElementById('searchInput');
        const suggestionsDiv = document.getElementById('suggestions');
        const loadingDiv = document.getElementById('loading');
        const resultsDiv = document.getElementById('results');
        let suggestionTimeout;

        // Handle search input with debouncing
        searchInput.addEventListener('input', function() {
            clearTimeout(suggestionTimeout);
            const query = this.value.trim();

            if (query.length < 2) {
                suggestionsDiv.style.display = 'none';
                return;
            }

            suggestionTimeout = setTimeout(() => {
                fetchSuggestions(query);
            }, 300);
        });

        // Handle search form submission
        document.getElementById('searchForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const query = searchInput.value.trim();
            if (query) {
                performSearch(query);
            }
        });

        // Fetch search suggestions
        async function fetchSuggestions(query) {
            try {
                const response = await fetch(`/api/suggestions?q=${encodeURIComponent(query)}`);
                const suggestions = await response.json();

                if (suggestions && suggestions.length > 0) {
                    displaySuggestions(suggestions);
                } else {
                    suggestionsDiv.style.display = 'none';
                }
            } catch (error) {
                console.error('Error fetching suggestions:', error);
            }
        }

        // Display search suggestions
        function displaySuggestions(suggestions) {
            suggestionsDiv.innerHTML = suggestions.map(suggestion =>
                `<div class="suggestion-item" style="padding: 8px; cursor: pointer; border-bottom: 1px solid #eee;" onclick="selectSuggestion('${suggestion}')">${suggestion}</div>`
            ).join('');
            suggestionsDiv.style.display = 'block';
        }

        // Select a suggestion
        function selectSuggestion(suggestion) {
            searchInput.value = suggestion;
            suggestionsDiv.style.display = 'none';
            performSearch(suggestion);
        }

        // Perform search
        async function performSearch(query) {
            loadingDiv.style.display = 'block';
            resultsDiv.innerHTML = '';
            suggestionsDiv.style.display = 'none';

            try {
                const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=20`);
                const data = await response.json();

                loadingDiv.style.display = 'none';
                displayResults(data);
            } catch (error) {
                loadingDiv.style.display = 'none';
                console.error('Error performing search:', error);
                resultsDiv.innerHTML = '<div class="alert alert-danger">Error performing search. Please try again.</div>';
            }
        }

        // Display search results
        function displayResults(data) {
            if (!data.results || data.results.length === 0) {
                resultsDiv.innerHTML = '<div class="alert alert-info">No results found.</div>';
                return;
            }

            let html = `<div class="stats alert alert-info">
                Found ${data.total_results} results in ${data.search_time.toFixed(2)} seconds
            </div>`;

            if (data.facets) {
                html += '<div class="facets"><div class="row">';
                Object.entries(data.facets).forEach(([facetType, facetData]) => {
                    if (Object.keys(facetData).length > 0) {
                        html += `<div class="col-md-3"><div class="facet-group">
                            <div class="facet-title">${facetType.replace('_', ' ').toUpperCase()}</div>`;
                        Object.entries(facetData).slice(0, 5).forEach(([value, count]) => {
                            html += `<div><small>${value}: ${count}</small></div>`;
                        });
                        html += '</div></div>';
                    }
                });
                html += '</div></div>';
            }

            data.results.forEach(result => {
                const complianceBadge = result.compliance_badge || '';
                const complianceScore = result.compliance ? result.compliance.compliance_score : null;
                const riskLevel = result.compliance ? result.compliance.risk_level : null;

                html += `<div class="result-card">
                    <div class="result-title">
                        <a href="/document/${result.doc_id}" onclick="trackAccess('${result.doc_id}')">
                            ${result.document.title || result.document.file_name}
                        </a>
                        ${complianceBadge ? `<span class="badge ${getComplianceBadgeClass(complianceBadge)} ms-2">${complianceBadge}</span>` : ''}
                    </div>
                    <div class="result-meta">
                        ${result.document.content_type || 'Document'} |
                        ${result.document.file_extension.toUpperCase()} |
                        ${result.document.file_size_human} |
                        Modified: ${new Date(result.document.date_modified).toLocaleDateString()}
                        ${complianceScore ? `| Compliance: ${complianceScore.toFixed(0)}%` : ''}
                        ${riskLevel ? `| Risk: ${riskLevel}` : ''}
                    </div>
                    ${result.snippet ? `<div class="result-snippet">${result.snippet}</div>` : ''}
                    <div class="result-score">
                        <small class="text-muted">Relevance Score: ${result.score.toFixed(3)}</small>
                    </div>
                </div>`;
            });

            resultsDiv.innerHTML = html;
        }

        // Track document access
        async function trackAccess(docId) {
            try {
                await fetch(`/api/track/${docId}`, { method: 'POST' });
            } catch (error) {
                console.error('Error tracking access:', error);
            }
        }

        // Hide suggestions when clicking outside
        document.addEventListener('click', function(e) {
            if (!suggestionsDiv.contains(e.target) && e.target !== searchInput) {
                suggestionsDiv.style.display = 'none';
            }
        });

        // Helper function for compliance badge styling
        function getComplianceBadgeClass(badge) {
            if (badge.includes('✅')) return 'badge-success';
            if (badge.includes('❌')) return 'badge-danger';
            if (badge.includes('⚠️')) return 'badge-warning';
            return 'badge-secondary';
        }

        // Initial search if query parameter exists
        const urlParams = new URLSearchParams(window.location.search);
        const initialQuery = urlParams.get('q');
        if (initialQuery) {
            searchInput.value = initialQuery;
            performSearch(initialQuery);
        }
    </script>
</body>
</html>
    """

    # Advanced search template
    advanced_search_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Search - Document Search Engine</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">📄 Advanced Document Search</a>
        </div>
    </nav>

    <div class="container mt-4">
        <h2>Advanced Search</h2>

        <form id="advancedSearchForm" class="mb-4">
            <div class="row">
                <div class="col-md-8">
                    <label for="query" class="form-label">Search Query</label>
                    <input type="text" class="form-control" id="query" name="query"
                           placeholder="Enter your search terms...">
                </div>
                <div class="col-md-4">
                    <label for="limit" class="form-label">Results Limit</label>
                    <select class="form-control" id="limit" name="limit">
                        <option value="10">10 results</option>
                        <option value="20" selected>20 results</option>
                        <option value="50">50 results</option>
                        <option value="100">100 results</option>
                    </select>
                </div>
            </div>

            <div class="row mt-3">
                <div class="col-md-6">
                    <label for="category" class="form-label">Category</label>
                    <select class="form-control" id="category" name="category">
                        <option value="">All Categories</option>
                        <option value="Invoice Processing">Invoice Processing</option>
                        <option value="Payment Processing">Payment Processing</option>
                        <option value="Vendor Management">Vendor Management</option>
                        <option value="Helpdesk Procedures">Helpdesk Procedures</option>
                        <option value="Period Close">Period Close</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label for="docType" class="form-label">Document Type</label>
                    <select class="form-control" id="docType" name="docType">
                        <option value="">All Types</option>
                        <option value="Procedure">Procedure</option>
                        <option value="Checklist">Checklist</option>
                        <option value="Guide">Guide</option>
                        <option value="Template">Template</option>
                        <option value="Reference">Reference</option>
                    </select>
                </div>
            </div>

            <div class="mt-3">
                <button type="submit" class="btn btn-primary">Search</button>
                <a href="/" class="btn btn-secondary">Back to Simple Search</a>
            </div>
        </form>

        <div id="results">
            <!-- Results will be displayed here -->
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('advancedSearchForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const params = new URLSearchParams();

            for (let [key, value] of formData.entries()) {
                if (value) {
                    params.append(key, value);
                }
            }

            const response = await fetch(`/api/search?${params}`);
            const data = await response.json();

            displayAdvancedResults(data);
        });

        function displayAdvancedResults(data) {
            const resultsDiv = document.getElementById('results');

            if (!data.results || data.results.length === 0) {
                resultsDiv.innerHTML = '<div class="alert alert-info">No results found.</div>';
                return;
            }

            let html = `<div class="alert alert-info">
                Found ${data.total_results} results in ${data.search_time.toFixed(2)} seconds
            </div>`;

            data.results.forEach(result => {
                html += `
                    <div class="card mb-3">
                        <div class="card-body">
                            <h5 class="card-title">
                                <a href="/document/${result.doc_id}">${result.document.title || result.document.file_name}</a>
                            </h5>
                            <p class="card-text">${result.snippet || 'No preview available'}</p>
                            <div class="mb-2">
                                <span class="badge bg-secondary">${result.document.content_type}</span>
                                <span class="badge bg-info">${result.document.file_extension.toUpperCase()}</span>
                                <span class="badge bg-success">${result.document.file_size_human}</span>
                            </div>
                            <small class="text-muted">
                                Modified: ${new Date(result.document.date_modified).toLocaleDateString()} |
                                Score: ${result.score.toFixed(3)}
                            </small>
                        </div>
                    </div>
                `;
            });

            resultsDiv.innerHTML = html;
        }
    </script>
</body>
</html>
    """

    # Document view template
    document_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Viewer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">📄 Document Search</a>
        </div>
    </nav>

    <div class="container mt-4" id="documentContent">
        <div class="text-center">
            <div class="spinner-border" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const docId = window.location.pathname.split('/').pop();

        // Load document details
        async function loadDocument() {
            try {
                const response = await fetch(`/api/document/${docId}`);
                const data = await response.json();

                if (data.document) {
                    displayDocument(data);
                } else {
                    document.getElementById('documentContent').innerHTML =
                        '<div class="alert alert-danger">Document not found</div>';
                }
            } catch (error) {
                console.error('Error loading document:', error);
                document.getElementById('documentContent').innerHTML =
                    '<div class="alert alert-danger">Error loading document</div>';
            }
        }

        function displayDocument(data) {
            const doc = data.document;
            let html = `
                <div class="row">
                    <div class="col-md-8">
                        <h2>${doc.title || doc.file_name}</h2>
                        <div class="mb-3">
                            <span class="badge bg-primary">${doc.content_type}</span>
                            <span class="badge bg-secondary">${doc.category}</span>
                            <span class="badge bg-info">${doc.document_type}</span>
                        </div>
                    `;

            if (data.snippet) {
                html += `<div class="alert alert-info">${data.snippet}</div>`;
            }

            html += `
                        <div class="document-meta mb-4">
                            <h5>Document Information</h5>
                            <table class="table table-sm">
                                <tr><td><strong>File Name:</strong></td><td>${doc.file_name}</td></tr>
                                <tr><td><strong>File Size:</strong></td><td>${doc.file_size_human}</td></tr>
                                <tr><td><strong>File Type:</strong></td><td>${doc.file_extension.toUpperCase()}</td></tr>
                                <tr><td><strong>Created:</strong></td><td>${new Date(doc.date_created).toLocaleString()}</td></tr>
                                <tr><td><strong>Modified:</strong></td><td>${new Date(doc.date_modified).toLocaleString()}</td></tr>
                                <tr><td><strong>Indexed:</strong></td><td>${new Date(doc.date_indexed).toLocaleString()}</td></tr>
            `;

            if (doc.author) {
                html += `<tr><td><strong>Author:</strong></td><td>${doc.author}</td></tr>`;
            }
            if (doc.version) {
                html += `<tr><td><strong>Version:</strong></td><td>${doc.version}</td></tr>`;
            }

            html += `</table></div>`;

            if (doc.tags && doc.tags.length > 0) {
                html += `
                    <div class="mb-4">
                        <h5>Keywords/Tags</h5>
                        ${doc.tags.map(tag => `<span class="badge bg-light text-dark me-1">${tag}</span>`).join('')}
                    </div>
                `;
            }

            html += `
                        <div class="mb-4">
                            <a href="/" class="btn btn-primary">Back to Search</a>
                            <button onclick="trackAccess()" class="btn btn-secondary">Mark as Viewed</button>
                        </div>
                    </div>

                    <div class="col-md-4">
                        <div class="card">
                            <div class="card-header">
                                <h5>Search Options</h5>
                            </div>
                            <div class="card-body">
                                <a href="/search?q=${encodeURIComponent(doc.title || doc.file_name)}" class="btn btn-outline-primary btn-sm">Search Similar</a>
                                <a href="/search?category=${encodeURIComponent(doc.category || '')}" class="btn btn-outline-secondary btn-sm">Browse Category</a>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            document.getElementById('documentContent').innerHTML = html;
        }

        async function trackAccess() {
            try {
                await fetch(`/api/track/${docId}`, { method: 'POST' });
                alert('Document access tracked!');
            } catch (error) {
                console.error('Error tracking access:', error);
            }
        }

        // Load document on page load
        loadDocument();
    </script>
</body>
</html>
    """

    # Write templates to files
    (templates_dir / 'search.html').write_text(search_html)
    (templates_dir / 'advanced_search.html').write_text(advanced_search_html)
    (templates_dir / 'document.html').write_text(document_html)
    (templates_dir / 'compliance_dashboard.html').write_text("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Compliance Monitoring Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .compliance-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem 0;
                margin-bottom: 2rem;
            }
            .stat-card {
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 1.5rem;
                text-align: center;
                transition: transform 0.2s;
            }
            .stat-card:hover {
                transform: translateY(-2px);
            }
            .stat-value {
                font-size: 2.5rem;
                font-weight: bold;
                margin-bottom: 0.5rem;
            }
            .stat-label {
                color: #666;
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .compliance-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 500;
                margin-left: 0.5rem;
            }
            .badge-compliant { background: #d4edda; color: #155724; }
            .badge-non-compliant { background: #f8d7da; color: #721c24; }
            .badge-partial { background: #fff3cd; color: #856404; }
            .activity-item {
                padding: 1rem;
                border-bottom: 1px solid #f0f0f0;
            }
            .activity-item:last-child {
                border-bottom: none;
            }
            .chart-container {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 2rem;
            }
        </style>
    </head>
    <body>
        <div class="compliance-header">
            <div class="container">
                <h1>📋 Compliance Monitoring Dashboard</h1>
                <p class="mb-0">Real-time compliance status and violation tracking</p>
            </div>
        </div>

        <div class="container">
            <div id="loading" class="text-center">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
            </div>

            <div id="dashboard-content" style="display: none;">
                <!-- Navigation -->
                <nav aria-label="breadcrumb" class="mb-4">
                    <ol class="breadcrumb">
                        <li class="breadcrumb-item"><a href="/">Search</a></li>
                        <li class="breadcrumb-item active">Compliance Dashboard</li>
                    </ol>
                </nav>

                <!-- Statistics Cards -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="stat-value text-info" id="total-documents">0</div>
                            <div class="stat-label">Total Documents</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="stat-value" id="compliance-rate">0%</div>
                            <div class="stat-label">Compliance Rate</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="stat-value text-danger" id="critical-violations">0</div>
                            <div class="stat-label">Critical Violations</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="stat-card">
                            <div class="stat-value text-warning" id="open-violations">0</div>
                            <div class="stat-label">Open Violations</div>
                        </div>
                    </div>
                </div>

                <!-- Charts and Activity -->
                <div class="row">
                    <div class="col-md-8">
                        <div class="chart-container">
                            <h4>📈 Compliance Trends (Last 30 Days)</h4>
                            <canvas id="trends-chart" width="400" height="200"></canvas>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="chart-container">
                            <h4>🕐 Recent Assessments</h4>
                            <div id="recent-assessments" style="max-height: 300px; overflow-y: auto;">
                                <!-- Recent assessments will be populated here -->
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Violations and Deadlines -->
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h4>⚠️ Active Violations</h4>
                            <div id="active-violations" style="max-height: 300px; overflow-y: auto;">
                                <!-- Active violations will be populated here -->
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="chart-container">
                            <h4>⏰ Upcoming Deadlines</h4>
                            <div id="upcoming-deadlines" style="max-height: 300px; overflow-y: auto;">
                                <!-- Deadlines will be populated here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="error" class="alert alert-danger" style="display: none;"></div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            let refreshTimer;

            document.addEventListener('DOMContentLoaded', function() {
                loadDashboardData();
                // Auto-refresh every 30 seconds
                refreshTimer = setInterval(loadDashboardData, 30000);
            });

            async function loadDashboardData() {
                try {
                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('dashboard-content').style.display = 'none';
                    document.getElementById('error').style.display = 'none';

                    const response = await fetch('/api/compliance/dashboard');
                    const data = await response.json();

                    if (data.error) {
                        showError(data.error);
                        return;
                    }

                    updateDashboard(data);
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('dashboard-content').style.display = 'block';

                } catch (error) {
                    showError('Failed to load dashboard data: ' + error.message);
                }
            }

            function updateDashboard(data) {
                // Update statistics
                const stats = data.overall_statistics;
                document.getElementById('total-documents').textContent = stats.total_documents || 0;
                document.getElementById('compliance-rate').textContent = stats.compliance_rate + '%';
                document.getElementById('critical-violations').textContent = stats.critical_violations || 0;
                document.getElementById('open-violations').textContent = stats.open_violations || 0;

                // Update recent assessments
                updateRecentAssessments(data.recent_assessments);

                // Update active violations
                updateActiveViolations(data.active_violations);

                // Update compliance trends chart
                updateTrendsChart(data.compliance_trends);

                // Update upcoming deadlines
                updateUpcomingDeadlines(data.upcoming_deadlines);
            }

            function updateRecentAssessments(assessments) {
                const container = document.getElementById('recent-assessments');

                if (!assessments || assessments.length === 0) {
                    container.innerHTML = '<p class="text-muted">No recent assessments</p>';
                    return;
                }

                container.innerHTML = assessments.map(assessment => `
                    <div class="activity-item">
                        <div>
                            <strong>${assessment.doc_title}</strong><br>
                            <small class="text-muted">
                                Score: ${assessment.compliance_score}% |
                                ${new Date(assessment.assessed_at).toLocaleDateString()}
                            </small>
                        </div>
                        <span class="badge ${getBadgeClass(assessment.overall_status)}">${assessment.overall_status}</span>
                    </div>
                `).join('');
            }

            function updateActiveViolations(violations) {
                const container = document.getElementById('active-violations');

                if (!violations || violations.length === 0) {
                    container.innerHTML = '<p class="text-muted">No active violations</p>';
                    return;
                }

                container.innerHTML = violations.slice(0, 10).map(violation => `
                    <div class="activity-item">
                        <div>
                            <strong>${violation.title}</strong><br>
                            <small class="text-muted">${violation.doc_title} | ${violation.days_open} days</small>
                        </div>
                        <span class="badge ${getBadgeClass(violation.severity)}">${violation.severity}</span>
                    </div>
                `).join('');
            }

            function updateTrendsChart(trendsData) {
                const ctx = document.getElementById('trends-chart').getContext('2d');

                if (window.trendsChart) {
                    window.trendsChart.destroy();
                }

                if (!trendsData.daily_scores || trendsData.daily_scores.length === 0) {
                    ctx.font = '16px Arial';
                    ctx.fillStyle = '#666';
                    ctx.textAlign = 'center';
                    ctx.fillText('No trend data available', ctx.canvas.width/2, ctx.canvas.height/2);
                    return;
                }

                window.trendsChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: trendsData.daily_scores.map(d => new Date(d.date).toLocaleDateString()),
                        datasets: [{
                            label: 'Compliance Score',
                            data: trendsData.daily_scores.map(d => d.average_score),
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                title: { display: true, text: 'Compliance Score (%)' }
                            }
                        },
                        plugins: { legend: { display: false } }
                    }
                });
            }

            function updateUpcomingDeadlines(deadlines) {
                const container = document.getElementById('upcoming-deadlines');

                if (!deadlines || deadlines.length === 0) {
                    container.innerHTML = '<p class="text-muted">No upcoming deadlines</p>';
                    return;
                }

                container.innerHTML = deadlines.slice(0, 5).map(deadline => `
                    <div class="activity-item">
                        <div>
                            <strong>${deadline.requirement_title}</strong><br>
                            <small class="text-muted">
                                Expires: ${new Date(deadline.expiry_date).toLocaleDateString()} (${deadline.days_until_expiry} days)
                            </small>
                        </div>
                        <span class="badge ${getBadgeClass(deadline.severity)}">${deadline.severity}</span>
                    </div>
                `).join('');
            }

            function getBadgeClass(status) {
                const statusMap = {
                    'compliant': 'badge-success',
                    'non_compliant': 'badge-danger',
                    'partial': 'badge-warning',
                    'critical': 'badge-danger',
                    'high': 'badge-warning',
                    'medium': 'badge-info',
                    'low': 'badge-success'
                };
                return statusMap[status] || 'badge-secondary';
            }

            function showError(message) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('dashboard-content').style.display = 'none';
                document.getElementById('error').textContent = message;
                document.getElementById('error').style.display = 'block';
            }

            // Cleanup timer on page unload
            window.addEventListener('beforeunload', function() {
                if (refreshTimer) clearInterval(refreshTimer);
            });
        </script>
    </body>
    </html>
    """)

    # Create basic error templates
    (templates_dir / '404.html').write_text("""
    <!DOCTYPE html>
    <html>
    <head><title>404 - Not Found</title></head>
    <body><h1>404 - Page Not Found</h1><a href="/">Go Home</a></body>
    </html>
    """)

    (templates_dir / '500.html').write_text("""
    <!DOCTYPE html>
    <html>
    <head><title>500 - Internal Server Error</title></head>
    <body><h1>500 - Internal Server Error</h1><a href="/">Go Home</a></body>
    </html>
    """)


def run_web_app(host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
    """Run the web application"""
    # Create templates if they don't exist
    if not Path('search_engine/templates').exists():
        create_templates()

    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_web_app(debug=True)