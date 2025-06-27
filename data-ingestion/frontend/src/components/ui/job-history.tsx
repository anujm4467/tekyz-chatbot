"use client"

import React, { useState, useEffect } from 'react'
import { Badge } from './badge'
import { 
  ChevronRight, 
  ChevronDown, 
  CheckCircle, 
  XCircle, 
  SkipForward,
  Activity,
  Circle
} from 'lucide-react'

interface JobMetrics {
  total_urls: number
  processed_urls: number
  successful_urls: number
  failed_urls: number
  total_vectors: number
  elapsed_time: number
}

interface PipelineStep {
  name: string
  order: number
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  progress: number
  items_processed: number
  items_total: number
  duration?: number
  error_message?: string
}

interface JobProgress {
  is_running: boolean
  progress: number
  current_step: string
  logs: string[]
  errors: Array<{ message?: string; timestamp?: string } | string>
  job_id: string
  metrics: JobMetrics
  eta: {
    eta_seconds: number | null
    eta_formatted: string | null
    estimated_completion: number | null
  }
  current_step_info: PipelineStep
  all_steps: PipelineStep[]
  overall_progress: number
  steps: PipelineStep[]
}

interface Job {
  job_id: string
  status: string
  start_time: number
  end_time?: number
  input_urls: string[]
  metrics?: JobMetrics
  error_message?: string
}

interface JobHistoryProps {
  apiUrl: string
}

export function JobHistory({ apiUrl }: JobHistoryProps) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedJob, setExpandedJob] = useState<string | null>(null)
  const [jobProgress, setJobProgress] = useState<Record<string, JobProgress>>({})
  const [loadingProgress, setLoadingProgress] = useState<Record<string, boolean>>({})
  const [stats, setStats] = useState<{
    statistics: {
      total_jobs: number
      completed_jobs: number
      failed_jobs: number
      running_jobs: number
      average_duration_seconds: number
    }
    active_jobs: number
    active_job_ids: string[]
  } | null>(null)

  const fetchJobs = async () => {
    try {
      const [jobsResponse, statsResponse] = await Promise.all([
        fetch(`${apiUrl}/pipeline/jobs`),
        fetch(`${apiUrl}/pipeline/stats`)
      ])
      
      const jobsData = await jobsResponse.json()
      const statsData = await statsResponse.json()
      
      setJobs(jobsData.jobs || [])
      setStats(statsData)
    } catch (error) {
      console.error('Failed to fetch job history:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchJobProgress = async (jobId: string) => {
    if (loadingProgress[jobId]) return

    setLoadingProgress(prev => ({ ...prev, [jobId]: true }))
    
    try {
      const response = await fetch(`${apiUrl}/pipeline/jobs/${jobId}/progress`)
      const progressData = await response.json()
      
      setJobProgress(prev => ({
        ...prev,
        [jobId]: progressData
      }))
    } catch (error) {
      console.error(`Failed to fetch progress for job ${jobId}:`, error)
    } finally {
      setLoadingProgress(prev => ({ ...prev, [jobId]: false }))
    }
  }

  const toggleJobExpansion = async (jobId: string) => {
    if (expandedJob === jobId) {
      setExpandedJob(null)
    } else {
      setExpandedJob(jobId)
      if (!jobProgress[jobId]) {
        await fetchJobProgress(jobId)
      }
    }
  }

  useEffect(() => {
    fetchJobs()
    const interval = setInterval(fetchJobs, 10000)
    return () => clearInterval(interval)
  }, [apiUrl])

  const formatDuration = (startTime: number, endTime?: number) => {
    const duration = (endTime || Date.now() / 1000) - startTime
    if (duration < 60) return `${Math.round(duration)}s`
    if (duration < 3600) return `${Math.round(duration / 60)}m`
    return `${Math.round(duration / 3600)}h`
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'running': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      case 'failed': return 'bg-red-500/20 text-red-400 border-red-500/30'
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500/30'
    }
  }

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'pending': return <Circle className="h-4 w-4 text-slate-400" />
      case 'running': return <Activity className="h-4 w-4 text-blue-400 animate-pulse" />
      case 'completed': return <CheckCircle className="h-4 w-4 text-green-400" />
      case 'failed': return <XCircle className="h-4 w-4 text-red-400" />
      case 'skipped': return <SkipForward className="h-4 w-4 text-yellow-400" />
      default: return <Circle className="h-4 w-4 text-slate-400" />
    }
  }

  const getStepStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'text-slate-400'
      case 'running': return 'text-blue-400'
      case 'completed': return 'text-green-400'
      case 'failed': return 'text-red-400'
      case 'skipped': return 'text-yellow-400'
      default: return 'text-slate-400'
    }
  }

  if (loading) {
    return (
      <div className="p-4 text-center text-slate-400">
        <Activity className="h-5 w-5 animate-spin mx-auto mb-2" />
        Loading job history...
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Statistics Summary */}
      {stats && (
        <div className="border-b border-slate-700 pb-3 mb-3">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-center">
              <div className="text-sm font-bold text-blue-400">{stats.statistics.total_jobs}</div>
              <div className="text-slate-500">Total</div>
            </div>
            <div className="text-center">
              <div className="text-sm font-bold text-green-400">{stats.statistics.completed_jobs}</div>
              <div className="text-slate-500">Completed</div>
            </div>
          </div>
        </div>
      )}

      {/* Job List */}
      {jobs.length === 0 ? (
        <div className="text-center text-slate-400 py-4">
          No jobs found
        </div>
      ) : (
        <div className="space-y-2">
          {jobs.map((job) => (
            <div key={job.job_id} className="border border-slate-700 rounded-lg overflow-hidden">
              {/* Job Header - Clickable */}
              <div
                onClick={() => toggleJobExpansion(job.job_id)}
                className="p-3 cursor-pointer hover:bg-slate-800/30 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {expandedJob === job.job_id ? (
                      <ChevronDown className="h-4 w-4 text-slate-400" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-slate-400" />
                    )}
                    <span className="font-mono text-xs text-slate-300">
                      {job.job_id.slice(0, 8)}...
                    </span>
                    <Badge className={`text-xs ${getStatusColor(job.status)}`}>
                      {job.status}
                    </Badge>
                  </div>
                  <div className="text-xs text-slate-400">
                    {formatDuration(job.start_time, job.end_time)}
                  </div>
                </div>

                <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <span className="text-slate-500">URLs:</span>{' '}
                    <span className="text-slate-300">{job.input_urls.length}</span>
                  </div>
                  {job.metrics && (
                    <>
                      <div>
                        <span className="text-slate-500">Success:</span>{' '}
                        <span className="text-green-400">{job.metrics.successful_urls}</span>
                      </div>
                      <div>
                        <span className="text-slate-500">Vectors:</span>{' '}
                        <span className="text-purple-400">{job.metrics.total_vectors}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Job Details - Expandable */}
              {expandedJob === job.job_id && (
                <div className="border-t border-slate-700 bg-slate-900/30">
                  {loadingProgress[job.job_id] ? (
                    <div className="p-4 text-center">
                      <Activity className="h-4 w-4 animate-spin mx-auto mb-2 text-blue-400" />
                      <div className="text-xs text-slate-400">Loading pipeline steps...</div>
                    </div>
                  ) : jobProgress[job.job_id] ? (
                    <div className="p-3">
                      <div className="text-xs font-medium text-slate-300 mb-3">Pipeline Steps:</div>
                      <div className="space-y-2">
                        {jobProgress[job.job_id].steps?.map((step) => (
                          <div key={step.name} className="flex items-center justify-between py-1">
                            <div className="flex items-center gap-2">
                              {getStepIcon(step.status)}
                              <span className={`text-xs font-medium ${getStepStatusColor(step.status)}`}>
                                {step.order}. {step.name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              {step.status === 'completed' && step.duration && (
                                <span className="text-xs text-slate-500">
                                  {step.duration < 1 ? '<1s' : `${Math.round(step.duration)}s`}
                                </span>
                              )}
                              {step.status === 'running' && (
                                <span className="text-xs text-blue-400">
                                  {step.progress}%
                                </span>
                              )}
                              {step.error_message && (
                                <span className="text-xs text-red-400 truncate max-w-32" title={step.error_message}>
                                  {step.error_message}
                                </span>
                              )}
                            </div>
                          </div>
                        )) || (
                          <div className="text-xs text-slate-500">No step details available</div>
                        )}
                      </div>
                      
                      {/* Overall Progress */}
                      {jobProgress[job.job_id].overall_progress !== undefined && (
                        <div className="mt-3 pt-2 border-t border-slate-700">
                          <div className="flex justify-between text-xs text-slate-400 mb-1">
                            <span>Overall Progress</span>
                            <span>{Math.round(jobProgress[job.job_id].overall_progress)}%</span>
                          </div>
                          <div className="w-full bg-slate-800 rounded-full h-1.5">
                            <div
                              className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
                              style={{ width: `${jobProgress[job.job_id].overall_progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="p-4 text-center text-xs text-slate-400">
                      Failed to load job details
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
} 