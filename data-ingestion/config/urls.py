"""
URL configurations for Tekyz website scraping
"""

# Manual URL list for Tekyz.com
TEKYZ_URLS = {
    'core_pages': [
        'https://tekyz.com/',
        'https://tekyz.com/about',
        'https://tekyz.com/services',
        'https://tekyz.com/portfolio',
        'https://tekyz.com/team',
        'https://tekyz.com/contact'
    ],
    'service_pages': [
        # These will be discovered dynamically from services page
        'https://tekyz.com/services/web-development',
        'https://tekyz.com/services/mobile-app-development',
        'https://tekyz.com/services/digital-marketing',
        'https://tekyz.com/services/ui-ux-design',
        'https://tekyz.com/services/e-commerce-development',
        'https://tekyz.com/services/cloud-solutions'
    ],
    'portfolio_pages': [
        # These will be discovered dynamically from portfolio page
    ],
    'blog_pages': [
        # These will be discovered dynamically if blog exists
    ]
}

# URL patterns to exclude
EXCLUDE_PATTERNS = [
    r'.*\.(jpg|jpeg|png|gif|pdf|doc|docx)$',
    r'.*\.(css|js|ico)$',
    r'.*/wp-admin/.*',
    r'.*/wp-content/.*',
    r'.*#.*',  # Anchor links
    r'.*\?.*',  # Query parameters (optional)
]

# URL patterns to include (whitelist)
INCLUDE_PATTERNS = [
    r'^https://tekyz\.com/?$',
    r'^https://tekyz\.com/[^/]+/?$',
    r'^https://tekyz\.com/[^/]+/[^/]+/?$',
]

def get_all_urls():
    """Get all predefined URLs as a flat list"""
    all_urls = []
    for category, urls in TEKYZ_URLS.items():
        all_urls.extend(urls)
    return all_urls

def get_core_urls():
    """Get only core page URLs"""
    return TEKYZ_URLS['core_pages']

def get_service_urls():
    """Get service page URLs"""
    return TEKYZ_URLS['service_pages']

def is_valid_tekyz_url(url: str) -> bool:
    """Check if URL is a valid Tekyz URL"""
    import re
    
    # Must be tekyz.com domain
    if not url.startswith('https://tekyz.com'):
        return False
    
    # Check exclude patterns
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, url, re.IGNORECASE):
            return False
    
    # Check include patterns
    for pattern in INCLUDE_PATTERNS:
        if re.match(pattern, url, re.IGNORECASE):
            return True
    
    return False 