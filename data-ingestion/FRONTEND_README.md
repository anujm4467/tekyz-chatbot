# Tekyz Data Pipeline - Modern Frontend

A beautiful, modern frontend built with **Next.js**, **shadcn/ui**, and **Tailwind CSS** for the Tekyz data ingestion pipeline.

## 🚀 Features

- **Modern UI**: Built with shadcn/ui components for a professional look
- **Dark Theme**: Beautiful dark mode with gradient backgrounds
- **Real-time Updates**: WebSocket integration for live pipeline status
- **File Upload**: Drag-and-drop file upload with support for .pdf, .docx, .txt
- **URL Processing**: Bulk URL input for web scraping
- **Progress Tracking**: Real-time progress bars and status indicators
- **Live Logs**: Real-time log streaming and error reporting
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Tech Stack

- **Next.js 15** - React framework with App Router
- **shadcn/ui** - High-quality, accessible UI components
- **Tailwind CSS** - Utility-first CSS framework
- **TypeScript** - Type-safe development
- **Lucide React** - Beautiful icons
- **React Dropzone** - File upload functionality
- **Axios** - HTTP client for API communication

## 📁 Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── globals.css          # Global styles and CSS variables
│   │   ├── layout.tsx           # Root layout with dark theme
│   │   └── page.tsx             # Main pipeline interface
│   ├── components/
│   │   └── ui/                  # shadcn/ui components
│   │       ├── button.tsx
│   │       ├── card.tsx
│   │       ├── progress.tsx
│   │       ├── badge.tsx
│   │       ├── textarea.tsx
│   │       ├── select.tsx
│   │       ├── dropdown-menu.tsx
│   │       ├── alert.tsx
│   │       └── separator.tsx
│   └── lib/
│       └── utils.ts             # Utility functions
├── components.json              # shadcn/ui configuration
├── tailwind.config.ts           # Tailwind CSS configuration
├── package.json
└── README.md
```

## 🚀 Quick Start

### Option 1: Start Everything (Recommended)

```bash
# Start both frontend and backend
./start_pipeline_ui.sh
```

### Option 2: Start Services Separately

**Start Backend:**

```bash
./start_backend.sh
```

**Start Frontend (in another terminal):**

```bash
./start_frontend.sh
```

### Option 3: Manual Start

**Backend:**

```bash
conda activate tekyz-data-ingestion
python backend_api.py
```

**Frontend:**

```bash
conda activate tekyz-data-ingestion
cd frontend
npm install  # First time only
npm run dev
```

## 🌐 Access Points

- **Frontend Application**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 🎯 Usage Guide

### 1. Upload Files

- Drag and drop files into the upload area
- Supported formats: `.pdf`, `.docx`, `.txt`
- Multiple files can be uploaded at once
- Files are displayed with size information
- Remove files using the X button

### 2. Add URLs

- Enter URLs in the textarea (one per line)
- URLs will be scraped for content
- Supports multiple URLs for batch processing

### 3. Start Pipeline

- Click "Start Pipeline" to begin processing
- Pipeline will process uploaded files and/or scrape URLs
- Real-time progress is shown with status updates

### 4. Monitor Progress

- **Status Panel**: Shows current pipeline state
- **Progress Bar**: Visual progress indicator
- **Current Step**: Shows which stage is running
- **Job ID**: Unique identifier for the current job

### 5. View Logs & Errors

- **Recent Logs**: Real-time log streaming
- **Recent Errors**: Error messages and alerts
- **WebSocket Status**: Connection indicator in header

### 6. Stop Pipeline

- Use the "Stop" button to halt a running pipeline
- The system will gracefully stop the current operation

## 🎨 UI Components

The interface uses shadcn/ui components for a consistent, professional look:

- **Cards**: Organize content into sections
- **Buttons**: Gradient buttons with loading states
- **Progress**: Real-time progress indicators
- **Badges**: Status and connection indicators
- **Alerts**: Error and warning messages
- **File Upload**: Drag-and-drop with visual feedback

## 🔌 API Integration

The frontend communicates with the FastAPI backend through:

- **REST API**: For pipeline control and status
- **WebSocket**: For real-time updates and logs
- **File Upload**: Multipart form data for file uploads

### Key Endpoints:

- `GET /pipeline/status` - Get current status
- `POST /pipeline/start` - Start pipeline with files/URLs
- `POST /pipeline/stop` - Stop running pipeline
- `WS /ws/pipeline` - WebSocket for real-time updates

## 🛠️ Development

### Adding New Components

```bash
# Add shadcn/ui components
cd frontend
npx shadcn@latest add [component-name]
```

### Available Components

- `button` - Various button styles
- `card` - Content containers
- `input` - Form inputs
- `textarea` - Multi-line text input
- `select` - Dropdown selections
- `progress` - Progress bars
- `badge` - Status indicators
- `alert` - Notifications
- `separator` - Visual dividers
- `dropdown-menu` - Context menus

### Customizing Theme

Edit `src/app/globals.css` to modify:

- Color variables
- Dark/light theme colors
- Border radius
- Custom scrollbar styling

## 🐛 Troubleshooting

### Common Issues

**Frontend won't start:**

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

**Backend connection issues:**

- Ensure backend is running on port 8000
- Check conda environment is activated
- Verify CORS settings in `backend_api.py`

**WebSocket connection fails:**

- Check browser console for connection errors
- Ensure backend WebSocket endpoint is accessible
- Verify firewall settings

**Upload not working:**

- Check file types are supported
- Ensure backend has write permissions to `data/raw/uploads`
- Verify multipart form data handling

### Development Mode

Enable verbose logging:

```bash
# Backend
LOG_LEVEL=DEBUG python backend_api.py

# Frontend (check browser console)
npm run dev
```

## 📝 License

Part of the Tekyz Data Pipeline project.
