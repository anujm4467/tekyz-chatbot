"use client";

import React, { useState, useEffect } from 'react';
import { Card } from './card';
import { Badge } from './badge';
import { Button } from './button';
import { ScrollArea } from './scroll-area';



interface PipelineStep {
  step_name: string;
  step_order: number;
  status: string;
  start_time: number | null;
  end_time: number | null;
  progress_percentage: number;
  items_total: number;
  items_processed: number;
  duration: number | null;
  error_message: string | null;
}

interface LogEntry {
  timestamp: number;
  level: string;
  message: string;
  step_name: string;
  formatted_time: string;
}

interface StepWithLogs {
  step: PipelineStep;
  logs: LogEntry[];
}

interface TimelineEntry {
  step_name: string;
  step_order: number;
  status: string;
  start_time: number | null;
  end_time: number | null;
  progress: number;
  start_formatted: string | null;
  end_formatted: string | null;
  duration: number | null;
}

interface MetricsEntry {
  timestamp: number;
  current_step: string;
  step_progress: number;
  processed_urls: number;
  total_urls: number;
  formatted_time: string;
}

interface JobSummary {
  total_duration: number | null;
  completed_steps: number;
  failed_steps: number;
  skipped_steps: number;
  total_steps: number;
}

interface VisualizationData {
  job: {
    job_id: string;
    status: string;
    start_time: number;
    end_time: number | null;
    input_urls: string[];
    error_message: string | null;
  };
  steps_with_logs: StepWithLogs[];
  timeline: TimelineEntry[];
  metrics_history: MetricsEntry[];
  summary: JobSummary;
}

interface PipelineVisualizationProps {
  jobId: string;
  onClose?: () => void;
}

const statusColors: Record<string, string> = {
  pending: 'bg-gray-100 text-gray-700',
  running: 'bg-blue-100 text-blue-700',
  completed: 'bg-green-100 text-green-700',
  failed: 'bg-red-100 text-red-700',
  skipped: 'bg-yellow-100 text-yellow-700'
};

const logLevelColors: Record<string, string> = {
  INFO: 'text-blue-600',
  WARNING: 'text-yellow-600',
  ERROR: 'text-red-600',
  DEBUG: 'text-gray-600'
};

export function PipelineVisualization({ jobId, onClose }: PipelineVisualizationProps) {
  const [data, setData] = useState<VisualizationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedStep, setSelectedStep] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'timeline' | 'logs' | 'metrics'>('overview');

  useEffect(() => {
    fetchVisualizationData();
  }, [jobId]);

  const fetchVisualizationData = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/pipeline/jobs/${jobId}/visualization`);
      if (!response.ok) {
        throw new Error('Failed to fetch visualization data');
      }
      const vizData = await response.json();
      setData(vizData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const formatDuration = (seconds: number | null): string => {
    if (!seconds) return 'N/A';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };



  if (loading) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-lg">Loading pipeline visualization...</div>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="p-6">
        <div className="text-center text-red-600">
          <h3 className="text-lg font-semibold mb-2">Error</h3>
          <p>{error}</p>
          <Button onClick={fetchVisualizationData} className="mt-4">
            Retry
          </Button>
        </div>
      </Card>
    );
  }

  if (!data) return null;

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Job Summary */}
      <Card className="p-4">
        <h3 className="font-semibold mb-3">Job Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Status:</span>
            <Badge className={`ml-2 ${statusColors[data.job.status] || 'bg-gray-100'}`}>
              {data.job.status}
            </Badge>
          </div>
          <div>
            <span className="text-gray-600">Duration:</span>
            <span className="ml-2 font-mono">
              {formatDuration(data.summary.total_duration)}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Steps:</span>
            <span className="ml-2">
              {data.summary.completed_steps}/{data.summary.total_steps}
            </span>
          </div>
          <div>
            <span className="text-gray-600">URLs:</span>
            <span className="ml-2">{data.job.input_urls.length}</span>
          </div>
        </div>
        {data.job.error_message && (
          <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded">
            <span className="text-red-600 text-sm">{data.job.error_message}</span>
          </div>
        )}
      </Card>

      {/* Steps Progress */}
      <Card className="p-4">
        <h3 className="font-semibold mb-3">Pipeline Steps</h3>
        <div className="space-y-3">
          {data.steps_with_logs.map(({ step }) => (
            <div key={step.step_name} className="flex items-center justify-between p-3 border rounded">
              <div className="flex items-center space-x-3">
                <span className="w-6 h-6 rounded-full bg-gray-100 flex items-center justify-center text-xs font-mono">
                  {step.step_order}
                </span>
                <div>
                  <div className="font-medium">{step.step_name}</div>
                  <div className="text-xs text-gray-600">
                    {step.items_processed}/{step.items_total} items
                  </div>
                </div>
              </div>
              <div className="flex items-center space-x-3">
                <div className="text-right text-xs text-gray-600">
                  {formatDuration(step.duration)}
                </div>
                <Badge className={statusColors[step.status]}>
                  {step.status}
                </Badge>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setSelectedStep(step.step_name)}
                >
                  View Logs
                </Button>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );

  const renderTimeline = () => (
    <Card className="p-4">
      <h3 className="font-semibold mb-3">Pipeline Timeline</h3>
      <div className="relative">
        {data.timeline.map((entry, index) => (
          <div key={entry.step_name} className="flex items-start space-x-4 pb-4">
            <div className="flex flex-col items-center">
              <div className={`w-3 h-3 rounded-full ${
                entry.status === 'completed' ? 'bg-green-500' :
                entry.status === 'failed' ? 'bg-red-500' :
                entry.status === 'running' ? 'bg-blue-500' :
                entry.status === 'skipped' ? 'bg-yellow-500' :
                'bg-gray-300'
              }`} />
              {index < data.timeline.length - 1 && (
                <div className="w-0.5 h-8 bg-gray-200 mt-1" />
              )}
            </div>
            <div className="flex-1">
              <div className="flex justify-between items-start">
                <div>
                  <h4 className="font-medium">{entry.step_name}</h4>
                  <div className="text-sm text-gray-600">
                    {entry.start_formatted} - {entry.end_formatted || 'In progress'}
                  </div>
                </div>
                <div className="text-right">
                  <Badge className={statusColors[entry.status]}>
                    {entry.status}
                  </Badge>
                  <div className="text-xs text-gray-600 mt-1">
                    {formatDuration(entry.duration)}
                  </div>
                </div>
              </div>
              {entry.progress > 0 && (
                <div className="mt-2">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${entry.progress}%` }}
                    />
                  </div>
                                     <div className="text-xs text-gray-600 mt-1">{(entry.progress || 0).toFixed(1)}%</div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );

  const renderLogs = () => {
    const logsToShow = selectedStep
      ? data.steps_with_logs.find(({ step }) => step.step_name === selectedStep)?.logs || []
      : data.steps_with_logs.flatMap(({ logs }) => logs).sort((a, b) => a.timestamp - b.timestamp);

    return (
      <Card className="p-4">
        <div className="flex justify-between items-center mb-3">
          <h3 className="font-semibold">
            {selectedStep ? `Logs for ${selectedStep}` : 'All Logs'}
          </h3>
          {selectedStep && (
            <Button size="sm" variant="outline" onClick={() => setSelectedStep(null)}>
              Show All Logs
            </Button>
          )}
        </div>
        <div className="h-96 overflow-y-auto">
          <div className="space-y-1 font-mono text-sm">
            {logsToShow.map((log, index) => (
              <div key={index} className="flex space-x-2 py-1">
                <span className="text-gray-500 w-20 flex-shrink-0">
                  {log.formatted_time}
                </span>
                <span className={`w-16 flex-shrink-0 ${logLevelColors[log.level] || 'text-gray-600'}`}>
                  {log.level}
                </span>
                {!selectedStep && (
                  <span className="text-blue-600 w-32 flex-shrink-0 truncate">
                    {log.step_name}
                  </span>
                )}
                <span className="flex-1">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      </Card>
    );
  };

  const renderMetrics = () => (
    <Card className="p-4">
      <h3 className="font-semibold mb-3">Metrics History</h3>
      <ScrollArea className="h-96">
        <div className="space-y-2">
          {data.metrics_history.map((metric, index) => (
            <div key={index} className="flex justify-between items-center p-2 border rounded text-sm">
              <div className="flex space-x-4">
                <span className="text-gray-600">{metric.formatted_time}</span>
                <span className="font-medium">{metric.current_step}</span>
              </div>
              <div className="flex space-x-4">
                                 <span>{(metric.step_progress || 0).toFixed(1)}% step progress</span>
                <span>{metric.processed_urls}/{metric.total_urls} URLs</span>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </Card>
  );

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold">Pipeline Visualization - {jobId}</h2>
        {onClose && (
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 border-b">
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'timeline', label: 'Timeline' },
          { id: 'logs', label: 'Logs' },
          { id: 'metrics', label: 'Metrics' }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as 'overview' | 'timeline' | 'logs' | 'metrics')}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg ${
              activeTab === tab.id
                ? 'bg-blue-50 text-blue-700 border-b-2 border-blue-700'
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="mt-4">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'timeline' && renderTimeline()}
        {activeTab === 'logs' && renderLogs()}
        {activeTab === 'metrics' && renderMetrics()}
      </div>
    </div>
  );
} 