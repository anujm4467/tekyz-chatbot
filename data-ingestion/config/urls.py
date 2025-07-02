"""
URL configurations for Tekyz website scraping
"""

# Comprehensive URL list for Tekyz.com based on site analysis
TEKYZ_URLS = {
    'core_pages': [
        'https://tekyz.com/',
        'https://tekyz.com/about',
        'https://tekyz.com/about-us',
        'https://tekyz.com/services',
        'https://tekyz.com/portfolio',
        'https://tekyz.com/team',
        'https://tekyz.com/contact',
        'https://tekyz.com/contact-us',
        'https://tekyz.com/careers',
        'https://tekyz.com/jobs',
    ],
    'service_pages': [
        # Core development services
        'https://tekyz.com/services/web-development',
        'https://tekyz.com/services/website-development',
        'https://tekyz.com/services/mobile-app-development',
        'https://tekyz.com/services/app-development',
        'https://tekyz.com/services/software-development',
        'https://tekyz.com/services/custom-software-development',
        
        # Design services
        'https://tekyz.com/services/ui-ux-design',
        'https://tekyz.com/services/web-design',
        'https://tekyz.com/services/graphic-design',
        'https://tekyz.com/services/logo-design',
        'https://tekyz.com/services/branding',
        
        # E-commerce services
        'https://tekyz.com/services/e-commerce-development',
        'https://tekyz.com/services/ecommerce-development',
        'https://tekyz.com/services/online-store-development',
        'https://tekyz.com/services/shopify-development',
        'https://tekyz.com/services/woocommerce-development',
        'https://tekyz.com/services/magento-development',
        
        # Digital marketing services
        'https://tekyz.com/services/digital-marketing',
        'https://tekyz.com/services/seo',
        'https://tekyz.com/services/search-engine-optimization',
        'https://tekyz.com/services/social-media-marketing',
        'https://tekyz.com/services/content-marketing',
        'https://tekyz.com/services/ppc',
        'https://tekyz.com/services/google-ads',
        'https://tekyz.com/services/facebook-ads',
        
        # Cloud and infrastructure
        'https://tekyz.com/services/cloud-solutions',
        'https://tekyz.com/services/cloud-migration',
        'https://tekyz.com/services/aws-services',
        'https://tekyz.com/services/azure-services',
        'https://tekyz.com/services/devops',
        'https://tekyz.com/services/hosting',
        'https://tekyz.com/services/server-management',
        
        # Consulting and support
        'https://tekyz.com/services/consulting',
        'https://tekyz.com/services/it-consulting',
        'https://tekyz.com/services/technology-consulting',
        'https://tekyz.com/services/maintenance',
        'https://tekyz.com/services/support',
        'https://tekyz.com/services/technical-support',
        
        # Emerging technologies
        'https://tekyz.com/services/ai-development',
        'https://tekyz.com/services/machine-learning',
        'https://tekyz.com/services/blockchain-development',
        'https://tekyz.com/services/iot-development',
        'https://tekyz.com/services/ar-vr-development',
    ],
    'portfolio_pages': [
        # Portfolio main sections
        'https://tekyz.com/portfolio/web-development',
        'https://tekyz.com/portfolio/mobile-apps',
        'https://tekyz.com/portfolio/e-commerce',
        'https://tekyz.com/portfolio/enterprise-solutions',
        'https://tekyz.com/portfolio/case-studies',
        'https://tekyz.com/portfolio/projects',
        'https://tekyz.com/work',
        'https://tekyz.com/case-studies',
        'https://tekyz.com/projects',
        
        # Industry-specific portfolios
        'https://tekyz.com/portfolio/healthcare',
        'https://tekyz.com/portfolio/finance',
        'https://tekyz.com/portfolio/education',
        'https://tekyz.com/portfolio/retail',
        'https://tekyz.com/portfolio/real-estate',
        'https://tekyz.com/portfolio/non-profit',
        'https://tekyz.com/portfolio/startups',
    ],
    'technology_pages': [
        # Frontend technologies
        'https://tekyz.com/technologies/react',
        'https://tekyz.com/technologies/angular',
        'https://tekyz.com/technologies/vue',
        'https://tekyz.com/technologies/javascript',
        'https://tekyz.com/technologies/html-css',
        'https://tekyz.com/technologies/next-js',
        
        # Backend technologies
        'https://tekyz.com/technologies/node-js',
        'https://tekyz.com/technologies/python',
        'https://tekyz.com/technologies/php',
        'https://tekyz.com/technologies/java',
        'https://tekyz.com/technologies/dotnet',
        'https://tekyz.com/technologies/ruby',
        'https://tekyz.com/technologies/go',
        
        # Mobile technologies
        'https://tekyz.com/technologies/react-native',
        'https://tekyz.com/technologies/flutter',
        'https://tekyz.com/technologies/ios',
        'https://tekyz.com/technologies/android',
        'https://tekyz.com/technologies/swift',
        'https://tekyz.com/technologies/kotlin',
        
        # Databases and cloud
        'https://tekyz.com/technologies/mysql',
        'https://tekyz.com/technologies/postgresql',
        'https://tekyz.com/technologies/mongodb',
        'https://tekyz.com/technologies/aws',
        'https://tekyz.com/technologies/azure',
        'https://tekyz.com/technologies/docker',
        'https://tekyz.com/technologies/kubernetes',
    ],
    'industry_pages': [
        'https://tekyz.com/industries/healthcare',
        'https://tekyz.com/industries/finance',
        'https://tekyz.com/industries/fintech',
        'https://tekyz.com/industries/education',
        'https://tekyz.com/industries/e-learning',
        'https://tekyz.com/industries/retail',
        'https://tekyz.com/industries/e-commerce',
        'https://tekyz.com/industries/real-estate',
        'https://tekyz.com/industries/logistics',
        'https://tekyz.com/industries/manufacturing',
        'https://tekyz.com/industries/travel',
        'https://tekyz.com/industries/hospitality',
        'https://tekyz.com/industries/media',
        'https://tekyz.com/industries/entertainment',
        'https://tekyz.com/industries/non-profit',
        'https://tekyz.com/industries/government',
        'https://tekyz.com/industries/startups',
    ],
    'blog_pages': [
        'https://tekyz.com/blog',
        'https://tekyz.com/insights',
        'https://tekyz.com/news',
        'https://tekyz.com/articles',
        'https://tekyz.com/resources',
        'https://tekyz.com/whitepapers',
        'https://tekyz.com/guides',
        'https://tekyz.com/tutorials',
        
        # Blog categories
        'https://tekyz.com/blog/web-development',
        'https://tekyz.com/blog/mobile-development',
        'https://tekyz.com/blog/digital-marketing',
        'https://tekyz.com/blog/technology-trends',
        'https://tekyz.com/blog/business-insights',
        'https://tekyz.com/blog/case-studies',
        'https://tekyz.com/blog/tutorials',
        'https://tekyz.com/blog/industry-news',
    ],
    'company_pages': [
        'https://tekyz.com/company',
        'https://tekyz.com/about-tekyz',
        'https://tekyz.com/our-story',
        'https://tekyz.com/mission',
        'https://tekyz.com/vision',
        'https://tekyz.com/values',
        'https://tekyz.com/leadership',
        'https://tekyz.com/management-team',
        'https://tekyz.com/founders',
        'https://tekyz.com/team-members',
        'https://tekyz.com/culture',
        'https://tekyz.com/office',
        'https://tekyz.com/locations',
        'https://tekyz.com/awards',
        'https://tekyz.com/certifications',
        'https://tekyz.com/partnerships',
        'https://tekyz.com/clients',
        'https://tekyz.com/testimonials',
        'https://tekyz.com/reviews',
    ],
    'career_pages': [
        'https://tekyz.com/careers',
        'https://tekyz.com/jobs',
        'https://tekyz.com/job-openings',
        'https://tekyz.com/current-openings',
        'https://tekyz.com/join-us',
        'https://tekyz.com/work-with-us',
        'https://tekyz.com/life-at-tekyz',
        'https://tekyz.com/benefits',
        'https://tekyz.com/employee-benefits',
        'https://tekyz.com/internships',
        'https://tekyz.com/graduate-program',
        'https://tekyz.com/remote-jobs',
        'https://tekyz.com/freelance-opportunities',
    ],
    'process_pages': [
        'https://tekyz.com/process',
        'https://tekyz.com/methodology',
        'https://tekyz.com/approach',
        'https://tekyz.com/how-we-work',
        'https://tekyz.com/development-process',
        'https://tekyz.com/project-management',
        'https://tekyz.com/agile-methodology',
        'https://tekyz.com/quality-assurance',
        'https://tekyz.com/testing',
        'https://tekyz.com/delivery',
        'https://tekyz.com/timeline',
        'https://tekyz.com/phases',
    ],
    'pricing_pages': [
        'https://tekyz.com/pricing',
        'https://tekyz.com/packages',
        'https://tekyz.com/plans',
        'https://tekyz.com/cost',
        'https://tekyz.com/rates',
        'https://tekyz.com/estimates',
        'https://tekyz.com/quote',
        'https://tekyz.com/get-quote',
        'https://tekyz.com/request-quote',
    ],
    'support_pages': [
        'https://tekyz.com/support',
        'https://tekyz.com/help',
        'https://tekyz.com/faq',
        'https://tekyz.com/documentation',
        'https://tekyz.com/knowledge-base',
        'https://tekyz.com/user-manual',
        'https://tekyz.com/privacy-policy',
        'https://tekyz.com/terms-of-service',
        'https://tekyz.com/terms-and-conditions',
        'https://tekyz.com/legal',
        'https://tekyz.com/cookies-policy',
        'https://tekyz.com/gdpr',
    ]
}

# More inclusive URL patterns to exclude
EXCLUDE_PATTERNS = [
    # File extensions
    r'.*\.(jpg|jpeg|png|gif|bmp|svg|webp|ico|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|tar|gz)$',
    r'.*\.(css|js|json|xml|txt|csv)$',
    r'.*\.(mp3|mp4|avi|mov|wmv|flv|webm|ogg)$',
    
    # Admin and system paths
    r'.*/wp-admin/.*',
    r'.*/wp-content/uploads/.*',
    r'.*/wp-includes/.*',
    r'.*/admin/.*',
    r'.*/dashboard/.*',
    r'.*/login.*',
    r'.*/register.*',
    r'.*/checkout.*',
    r'.*/cart.*',
    r'.*/account.*',
    r'.*/profile.*',
    
    # Search and filter URLs
    r'.*\?.*search.*',
    r'.*\?.*filter.*',
    r'.*\?.*page=.*',
    r'.*\?.*sort.*',
    r'.*\?.*category.*',
    
    # Anchor links and fragments
    r'.*#.*',
    
    # Duplicate content patterns
    r'.*\?print=.*',
    r'.*\?pdf=.*',
    r'.*/print/.*',
    
    # Feed and API URLs
    r'.*/feed/?$',
    r'.*/rss/?$',
    r'.*/atom/?$',
    r'.*/api/.*',
    r'.*/rest/.*',
    
    # Development and staging
    r'.*\.(dev|staging|test)\..*',
    r'.*/dev/.*',
    r'.*/staging/.*',
    r'.*/test/.*',
]

# More inclusive URL patterns to include
INCLUDE_PATTERNS = [
    # Main domain pages (any depth)
    r'^https://tekyz\.com/?$',
    r'^https://tekyz\.com/[^/\?#]+/?$',
    r'^https://tekyz\.com/[^/\?#]+/[^/\?#]+/?$',
    r'^https://tekyz\.com/[^/\?#]+/[^/\?#]+/[^/\?#]+/?$',
    r'^https://tekyz\.com/[^/\?#]+/[^/\?#]+/[^/\?#]+/[^/\?#]+/?$',
    
    # Subdomain support (if any)
    r'^https://[^.]+\.tekyz\.com/?$',
    r'^https://[^.]+\.tekyz\.com/[^/\?#]+/?$',
    r'^https://[^.]+\.tekyz\.com/[^/\?#]+/[^/\?#]+/?$',
]

# Common sitemap locations to check
SITEMAP_LOCATIONS = [
    '/sitemap.xml',
    '/sitemap_index.xml',
    '/sitemaps.xml',
    '/sitemap.xml.gz',
    '/wp-sitemap.xml',
    '/robots.txt',  # To extract sitemap URLs from robots.txt
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

def get_urls_by_category(category: str):
    """Get URLs for a specific category"""
    return TEKYZ_URLS.get(category, [])

def get_high_priority_urls():
    """Get high priority URLs for initial scraping"""
    high_priority = []
    high_priority.extend(TEKYZ_URLS['core_pages'])
    high_priority.extend(TEKYZ_URLS['service_pages'][:10])  # First 10 service pages
    high_priority.extend(TEKYZ_URLS['portfolio_pages'][:5])  # First 5 portfolio pages
    high_priority.extend(TEKYZ_URLS['company_pages'][:5])   # First 5 company pages
    return high_priority

def is_valid_tekyz_url(url: str) -> bool:
    """Check if URL is a valid Tekyz URL"""
    import re
    
    if not url or not isinstance(url, str):
        return False
    
    # Must be tekyz.com domain (including subdomains)
    if not re.match(r'^https?://(.*\.)?tekyz\.com', url):
        return False
    
    # Check exclude patterns first
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, url, re.IGNORECASE):
            return False
    
    # Check include patterns
    for pattern in INCLUDE_PATTERNS:
        if re.match(pattern, url, re.IGNORECASE):
            return True
    
    return False

def prioritize_urls(urls: list) -> list:
    """Prioritize URLs based on importance"""
    def get_priority_score(url: str) -> int:
        """Get priority score for URL (lower = higher priority)"""
        url_lower = url.lower()
        
        # Core pages have highest priority
        if any(core_url in url_lower for core_url in [u.lower() for u in TEKYZ_URLS['core_pages']]):
            return 1
        
        # Service pages
        if '/services' in url_lower:
            return 2
            
        # Company/about pages
        if any(keyword in url_lower for keyword in ['about', 'company', 'team', 'leadership']):
            return 3
            
        # Portfolio/work pages
        if any(keyword in url_lower for keyword in ['portfolio', 'work', 'case-studies', 'projects']):
            return 4
            
        # Technology pages
        if '/technologies' in url_lower:
            return 5
            
        # Industry pages
        if '/industries' in url_lower:
            return 6
            
        # Process/methodology pages
        if any(keyword in url_lower for keyword in ['process', 'methodology', 'approach']):
            return 7
            
        # Career pages
        if any(keyword in url_lower for keyword in ['careers', 'jobs', 'join']):
            return 8
            
        # Blog/content pages
        if any(keyword in url_lower for keyword in ['blog', 'news', 'insights', 'articles']):
            return 9
            
        # Support pages
        if any(keyword in url_lower for keyword in ['support', 'help', 'faq', 'documentation']):
            return 10
            
        # All other pages
        return 11
    
    # Sort by priority score, then alphabetically
    return sorted(urls, key=lambda x: (get_priority_score(x), x)) 