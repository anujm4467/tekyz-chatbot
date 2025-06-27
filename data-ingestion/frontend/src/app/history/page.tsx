"use client"

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import axios from 'axios'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  ArrowLeft,
  History, 
  Play, 
  CheckCircle, 
  XCircle, 
  Clock, 
  AlertCircle,
  Search,
  Filter,
  Database,
  FileText,
  Globe,
  Trash2,
  RefreshCw,
  Eye,
  Activity,
  Circle,
  BarChart3
} from 'lucide-react'
import { PipelineVisualization } from '@/components/ui/pipeline-visualization'

// Configure axios defaults
axios.defaults.baseURL = 'http://localhost:8000'

interface PipelineJob {
  id: string
  job_type: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  started_at?: string
  completed_at?: string
  urls?: string[]
  files?: string[]
  total_urls?: number
  processed_urls?: number
  successful_urls?: number
  failed_urls?: number
  total_chunks?: number
  total_embeddings?: number
  total_vectors?: number
  duplicate_count?: number
  error_message?: string
  duration?: number
  progress_percentage?: number
}

interface PipelineStats {
  total_jobs: number
  completed_jobs: number
  failed_jobs: number
  total_processing_time: number
  total_documents_processed: number
  total_vectors_created: number
}

export default function HistoryPage() {
  const router = useRouter()
  const [jobs, setJobs] = useState<PipelineJob[]>([])
  const [stats, setStats] = useState<PipelineStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [selectedJob, setSelectedJob] = useState<PipelineJob | null>(null)
  const [isDeleting, setIsDeleting] = useState<string | null>(null)
  const [showVisualization, setShowVisualization] = useState(false)

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000) // Refresh every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchData = async () => {
    try {
      const [jobsResponse, statsResponse] = await Promise.all([
        axios.get('/pipeline/jobs?limit=50'),
        axios.get('/pipeline/stats')
      ])
      setJobs(jobsResponse.data)
      setStats(statsResponse.data)
    } catch (error) {
      console.error('Failed to fetch history data:', error)
    } finally {
      setLoading(false)
    }
  }

  const deleteJob = async (jobId: string) => {
    if (!confirm('Are you sure you want to delete this job? This action cannot be undone.')) {
      return
    }

    setIsDeleting(jobId)
    try {
      await axios.delete(`/pipeline/jobs/${jobId}`)
      setJobs(jobs.filter(job => job.id !== jobId))
      if (selectedJob?.id === jobId) {
        setSelectedJob(null)
      }
    } catch (error) {
      console.error('Failed to delete job:', error)
      alert('Failed to delete job. Please try again.')
    } finally {
      setIsDeleting(null)
    }
  }

  const cleanupOldJobs = async () => {
    if (!confirm('Are you sure you want to cleanup old completed jobs? This will remove jobs older than 7 days.')) {
      return
    }

    try {
      await axios.post('/pipeline/cleanup')
      fetchData()
    } catch (error) {
      console.error('Failed to cleanup jobs:', error)
      alert('Failed to cleanup jobs. Please try again.')
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Clock className="h-4 w-4" />
      case 'running': return <Play className="h-4 w-4 animate-pulse" />
      case 'completed': return <CheckCircle className="h-4 w-4" />
      case 'failed': return <XCircle className="h-4 w-4" />
      case 'cancelled': return <AlertCircle className="h-4 w-4" />
      default: return <Circle className="h-4 w-4" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
      case 'running': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      case 'completed': return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'failed': return 'bg-red-500/20 text-red-400 border-red-500/30'
      case 'cancelled': return 'bg-gray-500/20 text-gray-400 border-gray-500/30'
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500/30'
    }
  }

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs}s`
  }

  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const filteredJobs = jobs.filter(job => {
    const matchesSearch = job.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         job.job_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         (job.urls && job.urls.some(url => url.toLowerCase().includes(searchTerm.toLowerCase())))
    
    const matchesStatus = statusFilter === 'all' || job.status === statusFilter
    
    return matchesSearch && matchesStatus
  })

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin text-blue-400 mx-auto mb-4" />
          <p className="text-slate-400">Loading pipeline history...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/50 backdrop-blur">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                onClick={() => router.push('/')}
                className="text-slate-400 hover:text-white"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to Pipeline
              </Button>
              <div className="flex items-center space-x-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-r from-purple-600 to-blue-600">
                  <History className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-white">Pipeline History</h1>
                  <p className="text-slate-400">Track and manage your data processing jobs</p>
                </div>
              </div>
            </div>
            
            <Button
              onClick={cleanupOldJobs}
              variant="outline"
              className="border-slate-600 text-slate-300 hover:bg-slate-800"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Cleanup Old Jobs
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Statistics Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8">
            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm">Total Jobs</p>
                    <p className="text-2xl font-bold text-white">{stats.total_jobs}</p>
                  </div>
                  <Database className="h-8 w-8 text-blue-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm">Completed</p>
                    <p className="text-2xl font-bold text-green-400">{stats.completed_jobs}</p>
                  </div>
                  <CheckCircle className="h-8 w-8 text-green-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm">Failed</p>
                    <p className="text-2xl font-bold text-red-400">{stats.failed_jobs}</p>
                  </div>
                  <XCircle className="h-8 w-8 text-red-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm">Documents</p>
                    <p className="text-2xl font-bold text-blue-400">{stats.total_documents_processed}</p>
                  </div>
                  <FileText className="h-8 w-8 text-blue-400" />
                </div>
              </CardContent>
            </Card>

            <Card className="bg-slate-900/50 border-slate-800">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-400 text-sm">Vectors</p>
                    <p className="text-2xl font-bold text-purple-400">{stats.total_vectors_created}</p>
                  </div>
                  <Activity className="h-8 w-8 text-purple-400" />
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Filters */}
        <Card className="bg-slate-900/50 border-slate-800 mb-8">
          <CardContent className="p-6">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 h-4 w-4" />
                  <Input
                    placeholder="Search jobs by ID, type, or URL..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 bg-slate-800/50 border-slate-700 text-white placeholder-slate-500"
                  />
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <Filter className="h-4 w-4 text-slate-400" />
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-40 bg-slate-800/50 border-slate-700 text-white">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-800 border-slate-700">
                    <SelectItem value="all">All Status</SelectItem>
                    <SelectItem value="pending">Pending</SelectItem>
                    <SelectItem value="running">Running</SelectItem>
                    <SelectItem value="completed">Completed</SelectItem>
                    <SelectItem value="failed">Failed</SelectItem>
                    <SelectItem value="cancelled">Cancelled</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Jobs List */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Jobs List */}
          <div className="space-y-4">
            <h2 className="text-xl font-semibold text-white mb-4">Recent Jobs ({filteredJobs.length})</h2>
            
            {filteredJobs.length === 0 ? (
              <Card className="bg-slate-900/50 border-slate-800">
                <CardContent className="p-12 text-center">
                  <History className="h-12 w-12 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-400">No jobs found matching your criteria</p>
                </CardContent>
              </Card>
            ) : (
              filteredJobs.map((job) => (
                <Card 
                  key={job.id} 
                  className={`bg-slate-900/50 border-slate-800 cursor-pointer transition-all hover:bg-slate-800/50 ${
                    selectedJob?.id === job.id ? 'ring-2 ring-blue-500' : ''
                  }`}
                  onClick={() => setSelectedJob(job)}
                >
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <Badge className={`${getStatusColor(job.status)} border`}>
                          {getStatusIcon(job.status)}
                          <span className="ml-1 capitalize">{job.status}</span>
                        </Badge>
                        <code className="text-xs bg-slate-800 px-2 py-1 rounded text-slate-300">
                          {job.id.slice(0, 8)}...
                        </code>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation()
                            setSelectedJob(job)
                          }}
                          className="text-slate-400 hover:text-white"
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation()
                            deleteJob(job.id)
                          }}
                          disabled={isDeleting === job.id}
                          className="text-red-400 hover:text-red-300"
                        >
                          {isDeleting === job.id ? (
                            <RefreshCw className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-sm">Type</span>
                        <span className="text-white text-sm capitalize">{job.job_type}</span>
                      </div>
                      
                      <div className="flex items-center justify-between">
                        <span className="text-slate-400 text-sm">Created</span>
                        <span className="text-white text-sm">{formatDateTime(job.created_at)}</span>
                      </div>
                      
                      {job.duration && (
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400 text-sm">Duration</span>
                          <span className="text-white text-sm">{formatDuration(job.duration)}</span>
                        </div>
                      )}

                      {job.progress_percentage !== undefined && (
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400 text-sm">Progress</span>
                          <span className="text-white text-sm">{job.progress_percentage}%</span>
                        </div>
                      )}

                      {job.total_vectors && (
                        <div className="flex items-center justify-between">
                          <span className="text-slate-400 text-sm">Vectors</span>
                          <span className="text-white text-sm">{job.total_vectors}</span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>

          {/* Job Details */}
          <div>
            <h2 className="text-xl font-semibold text-white mb-4">Job Details</h2>
            
            {selectedJob ? (
              <Card className="bg-slate-900/50 border-slate-800">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2 text-white">
                    {getStatusIcon(selectedJob.status)}
                    <span>Job {selectedJob.id.slice(0, 8)}...</span>
                    <Badge className={`${getStatusColor(selectedJob.status)} border ml-auto`}>
                      {selectedJob.status}
                    </Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Basic Info */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h4 className="text-sm font-medium text-slate-400 mb-2">Job Type</h4>
                      <p className="text-white capitalize">{selectedJob.job_type}</p>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-slate-400 mb-2">Status</h4>
                      <p className="text-white capitalize">{selectedJob.status}</p>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium text-slate-400 mb-2">Created</h4>
                      <p className="text-white">{formatDateTime(selectedJob.created_at)}</p>
                    </div>
                    {selectedJob.completed_at && (
                      <div>
                        <h4 className="text-sm font-medium text-slate-400 mb-2">Completed</h4>
                        <p className="text-white">{formatDateTime(selectedJob.completed_at)}</p>
                      </div>
                    )}
                  </div>

                  {/* Progress Info */}
                  {selectedJob.progress_percentage !== undefined && (
                    <div>
                      <h4 className="text-sm font-medium text-slate-400 mb-2">Progress</h4>
                      <div className="bg-slate-800 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full transition-all"
                          style={{ width: `${selectedJob.progress_percentage}%` }}
                        />
                      </div>
                      <p className="text-right text-sm text-slate-400 mt-1">{selectedJob.progress_percentage}%</p>
                    </div>
                  )}

                  {/* Processing Stats */}
                  {(selectedJob.total_urls || selectedJob.total_chunks || selectedJob.total_vectors) && (
                    <div>
                      <h4 className="text-sm font-medium text-slate-400 mb-3">Processing Statistics</h4>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        {selectedJob.total_urls && (
                          <div className="flex justify-between">
                            <span className="text-slate-400">URLs Processed:</span>
                            <span className="text-white">{selectedJob.processed_urls}/{selectedJob.total_urls}</span>
                          </div>
                        )}
                        {selectedJob.total_chunks && (
                          <div className="flex justify-between">
                            <span className="text-slate-400">Text Chunks:</span>
                            <span className="text-white">{selectedJob.total_chunks}</span>
                          </div>
                        )}
                        {selectedJob.total_embeddings && (
                          <div className="flex justify-between">
                            <span className="text-slate-400">Embeddings:</span>
                            <span className="text-white">{selectedJob.total_embeddings}</span>
                          </div>
                        )}
                        {selectedJob.total_vectors && (
                          <div className="flex justify-between">
                            <span className="text-slate-400">Vectors:</span>
                            <span className="text-white">{selectedJob.total_vectors}</span>
                          </div>
                        )}
                        {selectedJob.duplicate_count && (
                          <div className="flex justify-between">
                            <span className="text-slate-400">Duplicates:</span>
                            <span className="text-yellow-400">{selectedJob.duplicate_count}</span>
                          </div>
                        )}
                        {selectedJob.failed_urls && selectedJob.failed_urls > 0 && (
                          <div className="flex justify-between">
                            <span className="text-slate-400">Failed URLs:</span>
                            <span className="text-red-400">{selectedJob.failed_urls}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* URLs */}
                  {selectedJob.urls && selectedJob.urls.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-slate-400 mb-2">URLs ({selectedJob.urls.length})</h4>
                      <div className="bg-slate-950/50 rounded-lg p-3 max-h-32 overflow-y-auto">
                        {selectedJob.urls.map((url, index) => (
                          <div key={index} className="flex items-center space-x-2 text-sm py-1">
                            <Globe className="h-3 w-3 text-blue-400 flex-shrink-0" />
                            <span className="text-slate-300 truncate">{url}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Files */}
                  {selectedJob.files && selectedJob.files.length > 0 && (
                    <div>
                      <h4 className="text-sm font-medium text-slate-400 mb-2">Files ({selectedJob.files.length})</h4>
                      <div className="bg-slate-950/50 rounded-lg p-3 max-h-32 overflow-y-auto">
                        {selectedJob.files.map((file, index) => (
                          <div key={index} className="flex items-center space-x-2 text-sm py-1">
                            <FileText className="h-3 w-3 text-green-400 flex-shrink-0" />
                            <span className="text-slate-300 truncate">{file}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Error Message */}
                  {selectedJob.error_message && (
                    <div>
                      <h4 className="text-sm font-medium text-slate-400 mb-2">Error Details</h4>
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription className="text-sm">
                          {selectedJob.error_message}
                        </AlertDescription>
                      </Alert>
                    </div>
                  )}

                  {/* Job Actions */}
                  <div className="flex space-x-2 pt-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        navigator.clipboard.writeText(selectedJob.id)
                        alert('Job ID copied to clipboard')
                      }}
                      className="border-slate-600 text-slate-300 hover:bg-slate-800"
                    >
                      Copy Job ID
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowVisualization(true)}
                      className="border-purple-600 text-purple-300 hover:bg-purple-800/20"
                    >
                      <BarChart3 className="h-4 w-4 mr-2" />
                      View Pipeline Steps
                    </Button>
                    {selectedJob.status === 'failed' && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          if (confirm('Retry this job with the same parameters?')) {
                            // TODO: Implement retry functionality
                            alert('Retry functionality will be implemented')
                          }
                        }}
                        className="border-blue-600 text-blue-300 hover:bg-blue-800/20"
                      >
                        Retry Job
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="bg-slate-900/50 border-slate-800">
                <CardContent className="p-12 text-center">
                  <Eye className="h-12 w-12 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-400">Select a job to view details</p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </main>

      {/* Pipeline Visualization Modal */}
      {showVisualization && selectedJob && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-lg max-w-6xl w-full max-h-[90vh] overflow-hidden">
            <PipelineVisualization 
              jobId={selectedJob.id} 
              onClose={() => setShowVisualization(false)}
            />
          </div>
        </div>
      )}
    </div>
  )
} 