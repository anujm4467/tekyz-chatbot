"use client"

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './card'
import { Badge } from './badge'
import { Progress } from './progress'
import { CheckCircle2, Circle, XCircle, Clock, AlertTriangle } from 'lucide-react'

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

interface PipelineStepsProps {
  steps: PipelineStep[]
  currentStep?: PipelineStep | null
}

export function PipelineSteps({ steps, currentStep }: PipelineStepsProps) {
  const getStepIcon = (step: PipelineStep) => {
    const className = "w-5 h-5"
    
    switch (step.status) {
      case 'completed':
        return <CheckCircle2 className={`${className} text-green-500`} />
      case 'failed':
        return <XCircle className={`${className} text-red-500`} />
      case 'running':
        return <Clock className={`${className} text-blue-500 animate-spin`} />
      case 'skipped':
        return <AlertTriangle className={`${className} text-yellow-500`} />
      default:
        return <Circle className={`${className} text-gray-300`} />
    }
  }

  const getStepBadge = (step: PipelineStep) => {
    const variants: Record<string, 'default' | 'secondary' | 'destructive' | 'outline'> = {
      pending: 'outline',
      running: 'default',
      completed: 'secondary',
      failed: 'destructive',
      skipped: 'outline'
    }
    
    return (
      <Badge variant={variants[step.status] || 'outline'} className="text-xs">
        {step.status}
      </Badge>
    )
  }

  const formatStepName = (name: string | null | undefined) => {
    if (!name) return 'Unknown Step'
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')
  }

  const formatDuration = (duration?: number) => {
    if (!duration) return ''
    if (duration < 60) return `${Math.round(duration)}s`
    if (duration < 3600) return `${Math.round(duration / 60)}m ${Math.round(duration % 60)}s`
    return `${Math.round(duration / 3600)}h ${Math.round((duration % 3600) / 60)}m`
  }

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Pipeline Steps</CardTitle>
        {currentStep && (
          <div className="text-sm text-muted-foreground">
            Current: {formatStepName(currentStep.name)}
          </div>
        )}
      </CardHeader>
      
      <CardContent className="space-y-3">
        {steps.filter(step => step).map((step, index) => (
          <div key={step.name || `step-${index}`} className="space-y-2">
            {/* Step Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {getStepIcon(step)}
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm">
                      {step.order || 0}. {formatStepName(step.name)}
                    </span>
                    {getStepBadge(step)}
                  </div>
                  {step.error_message && (
                    <div className="text-xs text-red-500 mt-1">
                      {step.error_message}
                    </div>
                  )}
                </div>
              </div>
              
              <div className="text-right">
                {step.duration && (
                  <div className="text-xs text-muted-foreground">
                    {formatDuration(step.duration)}
                  </div>
                )}
                {(step.items_total || 0) > 0 && (
                  <div className="text-xs text-muted-foreground">
                    {step.items_processed || 0}/{step.items_total || 0}
                  </div>
                )}
              </div>
            </div>

            {/* Step Progress Bar */}
            {step.status === 'running' && (step.progress || 0) > 0 && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span>Progress</span>
                  <span>{(step.progress || 0).toFixed(1)}%</span>
                </div>
                <Progress value={step.progress || 0} className="h-2" />
              </div>
            )}

            {/* Connection Line */}
            {index < steps.length - 1 && (
              <div className="flex justify-start ml-2.5">
                <div className="w-0.5 h-4 bg-gray-200"></div>
              </div>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  )
} 