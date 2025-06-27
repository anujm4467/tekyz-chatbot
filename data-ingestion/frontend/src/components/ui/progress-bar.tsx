"use client"

import React from 'react'
import { Progress } from './progress'
import { Badge } from './badge'
import { Card, CardContent, CardHeader, CardTitle } from './card'

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

interface ProgressBarProps {
  progress: number
  currentStep: string
  metrics?: PipelineMetrics
  eta?: ETAInfo
  isRunning: boolean
  jobId?: string
}

export function EnhancedProgressBar({ 
  progress, 
  currentStep, 
  metrics, 
  eta, 
  isRunning, 
  jobId 
}: ProgressBarProps) {
  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`
    return `${Math.round(seconds / 3600)}h`
  }

  const formatRate = (rate: number | undefined) => {
    if (!rate || rate === 0) return '0/sec'
    if (rate < 1) return `${(rate * 60).toFixed(1)}/min`
    return `${rate.toFixed(1)}/sec`
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg">Pipeline Progress</CardTitle>
          {jobId && (
            <Badge variant="outline" className="text-xs">
              Job: {jobId.slice(0, 8)}...
            </Badge>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Main Progress Bar */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm font-medium">{currentStep}</span>
            <span className="text-sm text-muted-foreground">
              {(progress || 0).toFixed(1)}%
            </span>
          </div>
          <Progress value={progress} className="h-3" />
        </div>

        {/* ETA and Status */}
        {isRunning && (
          <div className="flex justify-between items-center text-sm">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-green-600 font-medium">Running</span>
            </div>
            {eta?.eta_formatted && (
              <span className="text-muted-foreground">
                ETA: {eta.eta_formatted}
              </span>
            )}
          </div>
        )}

        {/* Metrics Grid */}
        {metrics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2 border-t">
            <div className="text-center">
              <div className="text-lg font-bold text-blue-600">
                {metrics.processed_urls || 0}
              </div>
              <div className="text-xs text-muted-foreground">
                of {metrics.total_urls || 0} URLs
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-bold text-green-600">
                {metrics.successful_urls || 0}
              </div>
              <div className="text-xs text-muted-foreground">
                Successful
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-bold text-purple-600">
                {metrics.total_vectors || 0}
              </div>
              <div className="text-xs text-muted-foreground">
                Vectors Created
              </div>
            </div>
            
            <div className="text-center">
              <div className="text-lg font-bold text-orange-600">
                {formatRate(metrics.processing_rate)}
              </div>
              <div className="text-xs text-muted-foreground">
                Processing Rate
              </div>
            </div>
          </div>
        )}

        {/* Additional Metrics */}
        {metrics && (
          <div className="flex justify-between text-xs text-muted-foreground pt-2 border-t">
            <span>Chunks: {metrics.total_chunks || 0}</span>
            <span>Duplicates: {metrics.duplicate_count || 0}</span>
            <span>Runtime: {formatTime(metrics.elapsed_time || 0)}</span>
            {(metrics.failed_urls || 0) > 0 && (
              <span className="text-red-500">
                Failed: {metrics.failed_urls || 0}
              </span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
} 