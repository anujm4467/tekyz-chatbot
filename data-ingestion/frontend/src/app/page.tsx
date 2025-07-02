"use client"

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import axios from 'axios'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

import { Textarea } from '@/components/ui/textarea'

import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import { EnhancedProgressBar } from '@/components/ui/progress-bar'
import { JobHistory } from '@/components/ui/job-history'
import { PipelineSteps } from '@/components/ui/pipeline-steps'
import { 
  Upload, 
  Play, 
  Square, 
  Activity, 
  Zap, 
  FileText, 
  Globe, 
  CheckCircle, 
  AlertCircle,
  X,
  Clock,
  History,
  Database,
  RefreshCw
} from 'lucide-react'
import { useDropzone } from 'react-dropzone'

// Configure axios defaults
axios.defaults.baseURL = 'http://localhost:8000'

interface TekyzDataStatus {
  data_exists: boolean
  details: {
    exists: boolean
    reason: string
    vector_count: number
    tekyz_pages: number
    collection_info?: {
      name: string
      vector_size: number
      distance: string
    }
  }
}

interface PipelineMetrics {
  total_urls: number
  processed_urls: number
  successful_urls: number
  failed_urls: number
  total_chunks: number
  total_embeddings: number
  total_vectors: number
  duplicate_count: number
  processing_rate: number
  elapsed_time: number
  progress_percentage: number
}

interface ETAInfo {
  eta_seconds: number | null
  eta_formatted: string | null
  estimated_completion: number | null
}

interface PipelineStep {
  name: string | null | undefined
  order: number
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  progress: number
  items_processed: number
  items_total: number
  duration?: number
  error_message?: string
}

interface PipelineStatus {
  is_running: boolean
  progress: number
  current_step: string
  logs: string[]
  errors: Array<{ message?: string; timestamp?: string } | string>
  job_id?: string
  metrics?: PipelineMetrics
  eta?: ETAInfo
  current_step_info?: PipelineStep
  all_steps?: PipelineStep[]
}

export default function PipelinePage() {
  const router = useRouter()
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>({
    is_running: false,
    progress: 0,
    current_step: '',
    logs: [],
    errors: []
  })
  const [files, setFiles] = useState<File[]>([])
  const [urls, setUrls] = useState('')
  const [loading, setLoading] = useState(false)
  const [wsConnected, setWsConnected] = useState(false)
  const [activeTab, setActiveTab] = useState<'progress' | 'history'>('progress')
  
  // New state for tekyz data status
  const [tekyzDataStatus, setTekyzDataStatus] = useState<TekyzDataStatus | null>(null)
  const [checkingTekyzData, setCheckingTekyzData] = useState(false)
  const [scrapingTekyz, setScrapingTekyz] = useState(false)

  // WebSocket connection for real-time updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/pipeline')
    
    ws.onopen = () => {
      setWsConnected(true)
      console.log('WebSocket connected')
    }
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setPipelineStatus(prev => ({
        ...prev,
        ...data
      }))
    }
    
    ws.onclose = () => {
      setWsConnected(false)
      console.log('WebSocket disconnected')
    }
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setWsConnected(false)
    }
    
    return () => {
      ws.close()
    }
  }, [])

  // Poll for status updates with enhanced metrics
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const statusResponse = await axios.get('/pipeline/status')

        // Get current job progress if available
        let metrics: PipelineMetrics | undefined
        let eta: ETAInfo | undefined
        let current_step_info: PipelineStep | undefined
        let all_steps: PipelineStep[] | undefined
        
        if (statusResponse.data.job_id) {
          try {
            const progressResponse = await axios.get(`/pipeline/jobs/${statusResponse.data.job_id}/progress`)
            metrics = progressResponse.data.metrics
            eta = progressResponse.data.eta
            current_step_info = progressResponse.data.current_step_info
            all_steps = progressResponse.data.all_steps
          } catch (error) {
            console.warn('Failed to fetch job progress:', error)
          }
        }

        setPipelineStatus(prev => ({
          ...prev,
          ...statusResponse.data,
          metrics,
          eta,
          current_step_info,
          all_steps
        }))
      } catch (error) {
        console.error('Failed to fetch status:', error)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  // Dropzone configuration
  const onDrop = (acceptedFiles: File[]) => {
    setFiles(prev => [...prev, ...acceptedFiles])
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt']
    },
    multiple: true
  })

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index))
  }

  const startPipeline = async () => {
    setLoading(true)
    try {
      const formData = new FormData()
      files.forEach(file => {
        formData.append('files', file)
      })
      
      const urlList = urls.split('\n').filter(url => url.trim())
      urlList.forEach(url => {
        formData.append('urls', url.trim())
      })

      await axios.post('/pipeline/start', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      
      console.log('Pipeline started successfully')
    } catch (error) {
      console.error('Failed to start pipeline:', error)
    } finally {
      setLoading(false)
    }
  }

  const stopPipeline = async () => {
    try {
      await axios.post('/pipeline/stop')
      console.log('Pipeline stopped')
    } catch (error) {
      console.error('Failed to stop pipeline:', error)
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // Check tekyz data status on component mount
  useEffect(() => {
    checkTekyzDataStatus()
  }, [])

  const checkTekyzDataStatus = async () => {
    setCheckingTekyzData(true)
    try {
      const response = await axios.get('/pipeline/tekyz-data-status')
      setTekyzDataStatus(response.data)
    } catch (error) {
      console.error('Failed to check tekyz data status:', error)
    } finally {
      setCheckingTekyzData(false)
    }
  }

  const startTekyzScraping = async () => {
    setScrapingTekyz(true)
    try {
      const response = await axios.post('/pipeline/scrape-tekyz')
      if (response.data.success) {
        // Pipeline will be tracked via WebSocket
        console.log('Tekyz scraping started:', response.data.job_id)
      } else {
        console.error('Failed to start tekyz scraping:', response.data.error)
      }
    } catch (error) {
      console.error('Error starting tekyz scraping:', error)
    } finally {
      setScrapingTekyz(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/50 backdrop-blur supports-[backdrop-filter]:bg-slate-950/50">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-r from-blue-600 to-violet-600">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">Tekyz Data Pipeline</h1>
                <p className="text-slate-400">Advanced data ingestion and processing platform</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* History Link */}
              <Button
                variant="outline"
                onClick={() => router.push('/history')}
                className="border-slate-600 text-slate-300 hover:bg-slate-800"
              >
                <History className="h-4 w-4 mr-2" />
                View History
              </Button>
              
              {/* Connection Status */}
              <Badge variant={wsConnected ? "default" : "destructive"} className="flex items-center space-x-2">
                <div className={`h-2 w-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'}`} />
                <span>{wsConnected ? 'Connected' : 'Disconnected'}</span>
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* Enhanced Pipeline Status with Tabs */}
          <div className="lg:col-span-1">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center space-x-2 text-white">
                    <Activity className="h-5 w-5" />
                    <span>Pipeline Monitor</span>
                  </CardTitle>
                  
                  {/* Tab Navigation */}
                  <div className="flex bg-slate-800 rounded-lg p-1">
                    <button
                      onClick={() => setActiveTab('progress')}
                      className={`px-3 py-1 text-xs rounded transition-colors ${
                        activeTab === 'progress'
                          ? 'bg-blue-600 text-white'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      <Activity className="h-3 w-3 inline mr-1" />
                      Live
                    </button>
                    <button
                      onClick={() => setActiveTab('history')}
                      className={`px-3 py-1 text-xs rounded transition-colors ${
                        activeTab === 'history'
                          ? 'bg-blue-600 text-white'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      <History className="h-3 w-3 inline mr-1" />
                      History
                    </button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                {activeTab === 'progress' ? (
                  <div className="p-6">
                    <EnhancedProgressBar
                      progress={pipelineStatus.progress || 0}
                      currentStep={pipelineStatus.current_step || 'Idle'}
                      metrics={pipelineStatus.metrics}
                      eta={pipelineStatus.eta}
                      isRunning={pipelineStatus.is_running}
                      jobId={pipelineStatus.job_id}
                    />
                  </div>
                ) : (
                  <div className="max-h-96 overflow-y-auto">
                    <JobHistory apiUrl="http://localhost:8000" />
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Pipeline Controls */}
          <div className="lg:col-span-3">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardHeader>
                <CardTitle className="text-white">Pipeline Controls</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                
                {/* Tekyz Data Status Card */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Database className="h-5 w-5" />
                      Tekyz.com Data Status
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {checkingTekyzData ? (
                      <div className="flex items-center gap-2">
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        <span>Checking database...</span>
                      </div>
                    ) : tekyzDataStatus ? (
                      <div className="space-y-3">
                        <div className="flex items-center gap-2">
                          {tekyzDataStatus.data_exists ? (
                            <>
                              <CheckCircle className="h-5 w-5 text-green-500" />
                              <span className="text-green-700">Tekyz.com data already exists in database</span>
                            </>
                          ) : (
                            <>
                              <AlertCircle className="h-5 w-5 text-yellow-500" />
                              <span className="text-yellow-700">No tekyz.com data found in database</span>
                            </>
                          )}
                        </div>
                        
                        {tekyzDataStatus.details && (
                          <div className="bg-gray-50 p-3 rounded-lg space-y-2">
                            <p className="text-sm text-gray-600">{tekyzDataStatus.details.reason}</p>
                            {tekyzDataStatus.details.vector_count > 0 && (
                              <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                  <span className="font-medium">Total Vectors:</span> {tekyzDataStatus.details.vector_count}
                                </div>
                                <div>
                                  <span className="font-medium">Tekyz Pages:</span> {tekyzDataStatus.details.tekyz_pages}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                        
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={checkTekyzDataStatus}
                            disabled={checkingTekyzData}
                          >
                            <RefreshCw className={`h-4 w-4 mr-2 ${checkingTekyzData ? 'animate-spin' : ''}`} />
                            Refresh Status
                          </Button>
                          
                          {!tekyzDataStatus.data_exists && (
                            <Button
                              onClick={startTekyzScraping}
                              disabled={scrapingTekyz || pipelineStatus.is_running}
                              size="sm"
                            >
                              <Globe className="h-4 w-4 mr-2" />
                              {scrapingTekyz ? 'Starting...' : 'Scrape Tekyz.com'}
                            </Button>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="text-gray-500">Unable to check database status</div>
                    )}
                  </CardContent>
                </Card>

                {/* File Upload Card */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Upload className="h-5 w-5" />
                      Upload Documents
                      <Badge variant="outline">Primary Input</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div
                      {...getRootProps()}
                      className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
                        isDragActive 
                          ? 'border-blue-500 bg-blue-500/10' 
                          : 'border-slate-600 hover:border-slate-500 hover:bg-slate-800/30'
                      }`}
                    >
                      <input {...getInputProps()} />
                      <Upload className="mx-auto h-12 w-12 text-slate-400 mb-4" />
                      <div className="text-slate-300 mb-2">
                        <span className="font-medium text-blue-400">Click to upload</span> or drag and drop
                      </div>
                      <p className="text-sm text-slate-500">Supports .docx, .pdf, .txt files</p>
                    </div>

                    {/* File List */}
                    {files.length > 0 && (
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium text-slate-300">Selected Files</h4>
                        {files.map((file, index) => (
                          <div key={index} className="flex items-center justify-between bg-slate-800/50 rounded-lg p-3">
                            <div className="flex items-center space-x-3">
                              <FileText className="h-4 w-4 text-blue-400" />
                              <span className="text-white text-sm">{file.name}</span>
                              <span className="text-slate-400 text-xs">({formatFileSize(file.size)})</span>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => removeFile(index)}
                              className="text-red-400 hover:text-red-300 hover:bg-red-400/10"
                            >
                              <X className="h-4 w-4" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Separator className="bg-slate-800" />
                
                {/* URL Input Card */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Globe className="h-5 w-5" />
                      URLs (Optional)
                      <Badge variant="secondary">Secondary Input</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Alert>
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        <strong>Note:</strong> If documents are uploaded, URLs will be skipped. 
                        Documents take priority in the processing pipeline.
                      </AlertDescription>
                    </Alert>
                    
                    <div className="space-y-2">
                      <label className="text-sm font-medium">URLs to scrape (one per line):</label>
                      <Textarea
                        placeholder="https://example.com/page1&#10;https://example.com/page2"
                        value={urls}
                        onChange={(e) => setUrls(e.target.value)}
                        rows={4}
                        className="font-mono text-sm"
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Action Buttons */}
                <div className="flex space-x-4">
                  <Button
                    onClick={startPipeline}
                    disabled={loading || pipelineStatus.is_running || (files.length === 0 && !urls.trim())}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-700 hover:to-violet-700"
                  >
                    {loading ? (
                      <>
                        <Clock className="mr-2 h-4 w-4 animate-spin" />
                        Starting...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Start Pipeline
                      </>
                    )}
                  </Button>
                  
                  <Button
                    onClick={stopPipeline}
                    disabled={!pipelineStatus.is_running}
                    variant="destructive"
                    className="px-8"
                  >
                    <Square className="mr-2 h-4 w-4" />
                    Stop
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Pipeline Steps and Logs */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
          {/* Pipeline Steps */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardContent className="p-6">
              {pipelineStatus.all_steps && pipelineStatus.all_steps.length > 0 ? (
                <PipelineSteps 
                  steps={pipelineStatus.all_steps} 
                  currentStep={pipelineStatus.current_step_info}
                />
              ) : (
                <div className="text-slate-500 text-center py-8">
                  <Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  No pipeline steps available
                </div>
              )}
            </CardContent>
          </Card>

          {/* Recent Logs */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2 text-white">
                <FileText className="h-5 w-5" />
                <span>Recent Logs</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-slate-950/50 rounded-lg p-4 max-h-64 overflow-y-auto">
                {pipelineStatus.logs && pipelineStatus.logs.length > 0 ? (
                  <div className="space-y-1">
                    {pipelineStatus.logs.slice(-10).map((log, index) => (
                      <div key={index} className="text-sm text-slate-300 font-mono">
                        {log}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-slate-500 text-center py-8">
                    <FileText className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    No logs available
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Recent Errors */}
          <Card className="bg-slate-900/50 border-slate-800">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2 text-white">
                <AlertCircle className="h-5 w-5" />
                <span>Recent Errors</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-slate-950/50 rounded-lg p-4 max-h-64 overflow-y-auto">
                {pipelineStatus.errors && pipelineStatus.errors.length > 0 ? (
                  <div className="space-y-2">
                    {pipelineStatus.errors.slice(-5).map((error, index) => (
                      <Alert key={index} variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription className="text-sm">
                          {typeof error === 'string' ? error : error.message || 'Unknown error'}
                        </AlertDescription>
                      </Alert>
                    ))}
                  </div>
                ) : (
                  <div className="text-slate-500 text-center py-8">
                    <CheckCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    No errors
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
