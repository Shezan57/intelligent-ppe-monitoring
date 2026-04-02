/**
 * History Table Component
 * 
 * Displays violation history with filtering and pagination.
 * Supports both "All Detections" and "Verified Only" (Judge-confirmed) modes.
 */

import { useState, useEffect, useCallback } from 'react'
import { getHistory, getHistorySummary, getVerifiedViolations } from '../api/client'

function HistoryTable() {
    // State
    const [violations, setViolations] = useState([])
    const [summary, setSummary] = useState(null)
    const [isLoading, setIsLoading] = useState(true)
    const [error, setError] = useState(null)

    // Mode toggle: "all" or "verified"
    const [mode, setMode] = useState('all')

    // Filters
    const [dateFilter, setDateFilter] = useState('')
    const [typeFilter, setTypeFilter] = useState('')

    // Pagination
    const [page, setPage] = useState(1)
    const [totalPages, setTotalPages] = useState(1)
    const pageSize = 10

    // Fetch violations based on mode
    const fetchViolations = useCallback(async () => {
        setIsLoading(true)
        setError(null)

        try {
            if (mode === 'verified') {
                // Fetch Judge-confirmed violations
                const params = {
                    limit: pageSize,
                    offset: (page - 1) * pageSize,
                }
                if (typeFilter) params.violation_type = typeFilter

                const data = await getVerifiedViolations(params)
                setViolations(data.violations || [])
                setTotalPages(Math.ceil((data.total_count || 0) / pageSize))
            } else {
                // Fetch all detections
                const params = { page, page_size: pageSize }
                if (dateFilter) {
                    params.date_from = dateFilter
                    params.date_to = dateFilter
                }
                if (typeFilter) params.violation_type = typeFilter

                const data = await getHistory(params)
                setViolations(data.violations || [])
                setTotalPages(Math.ceil((data.total_count || data.total || 0) / pageSize))
            }
        } catch (err) {
            console.error('Failed to fetch history:', err)
            setError('Failed to load violation history')
        } finally {
            setIsLoading(false)
        }
    }, [page, dateFilter, typeFilter, mode])

    // Fetch summary
    const fetchSummary = useCallback(async () => {
        try {
            const data = await getHistorySummary(7)
            setSummary(data)
        } catch (err) {
            console.error('Failed to fetch summary:', err)
        }
    }, [])

    // Load data on mount and filter/mode changes
    useEffect(() => {
        fetchViolations()
    }, [fetchViolations])

    useEffect(() => {
        fetchSummary()
    }, [fetchSummary])

    // Reset to page 1 when filters or mode change
    useEffect(() => {
        setPage(1)
    }, [dateFilter, typeFilter, mode])

    // Format timestamp
    const formatTime = (timestamp) => {
        if (!timestamp) return '-'
        const date = new Date(timestamp)
        return date.toLocaleString()
    }

    // Get violation type badge
    const getViolationBadge = (type) => {
        const badges = {
            'no_helmet': { label: 'No Helmet', className: 'badge badge--warning' },
            'no_vest': { label: 'No Vest', className: 'badge badge--warning' },
            'both_missing': { label: 'Both Missing', className: 'badge badge--violation' }
        }
        const badge = badges[type] || { label: type, className: 'badge' }
        return <span className={badge.className}>{badge.label}</span>
    }

    return (
        <div className="history-container">
            {/* Summary Cards */}
            {summary && (
                <div className="summary-cards">
                    <div className="card summary-card">
                        <div className="summary-card__value">{summary.total_violations || 0}</div>
                        <div className="summary-card__label">Total Violations (7 days)</div>
                    </div>
                    <div className="card summary-card">
                        <div className="summary-card__value">{summary.compliance_rate?.toFixed(1) || 0}%</div>
                        <div className="summary-card__label">Compliance Rate</div>
                    </div>
                    <div className="card summary-card">
                        <div className="summary-card__value">{summary.no_helmet_count || 0}</div>
                        <div className="summary-card__label">No Helmet</div>
                    </div>
                    <div className="card summary-card">
                        <div className="summary-card__value">{summary.no_vest_count || 0}</div>
                        <div className="summary-card__label">No Vest</div>
                    </div>
                </div>
            )}

            {/* Mode Toggle + Filters */}
            <div className="card filters-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <h4>🔍 Filters</h4>
                    <div style={{ display: 'flex', gap: '0.25rem', background: 'var(--bg-secondary)', borderRadius: '8px', padding: '2px' }}>
                        <button
                            className={`btn btn--sm ${mode === 'all' ? 'btn--primary' : 'btn--ghost'}`}
                            onClick={() => setMode('all')}
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.75rem', borderRadius: '6px' }}
                        >
                            All Detections
                        </button>
                        <button
                            className={`btn btn--sm ${mode === 'verified' ? 'btn--primary' : 'btn--ghost'}`}
                            onClick={() => setMode('verified')}
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.75rem', borderRadius: '6px' }}
                        >
                            ⚖️ Verified Only
                        </button>
                    </div>
                </div>

                <div className="filters-row">
                    {mode === 'all' && (
                        <div className="filter-group">
                            <label>Date</label>
                            <input
                                type="date"
                                value={dateFilter}
                                onChange={(e) => setDateFilter(e.target.value)}
                                className="input"
                            />
                        </div>
                    )}
                    <div className="filter-group">
                        <label>Violation Type</label>
                        <select
                            value={typeFilter}
                            onChange={(e) => setTypeFilter(e.target.value)}
                            className="input"
                        >
                            <option value="">All Types</option>
                            <option value="no_helmet">No Helmet</option>
                            <option value="no_vest">No Vest</option>
                            <option value="both_missing">Both Missing</option>
                        </select>
                    </div>
                    <button
                        className="btn btn--secondary"
                        onClick={() => {
                            setDateFilter('')
                            setTypeFilter('')
                        }}
                    >
                        Clear Filters
                    </button>
                </div>
            </div>

            {/* Table */}
            <div className="card">
                <h4>
                    {mode === 'verified'
                        ? '⚖️ Verified Violations (Judge-Confirmed)'
                        : '📋 Violation History'}
                </h4>

                {isLoading ? (
                    <div className="loading-container">
                        <span className="loading-spinner"></span>
                        <p>Loading violations...</p>
                    </div>
                ) : error ? (
                    <div className="error-message">
                        <p>❌ {error}</p>
                        <button className="btn btn--primary" onClick={fetchViolations}>
                            Retry
                        </button>
                    </div>
                ) : violations.length === 0 ? (
                    <div className="empty-state">
                        <p>{mode === 'verified' ? 'No verified violations found. Run the pipeline first.' : 'No violations found'}</p>
                    </div>
                ) : (
                    <>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    {mode === 'verified' && <th>Person ID</th>}
                                    {mode === 'all' && <th>Site</th>}
                                    <th>{mode === 'verified' ? 'Zone' : 'Camera'}</th>
                                    <th>Type</th>
                                    <th>Path</th>
                                    <th>{mode === 'verified' ? 'Judge Conf.' : 'Confidence'}</th>
                                    {mode === 'verified' && <th>SAM Time</th>}
                                </tr>
                            </thead>
                            <tbody>
                                {violations.map((v, idx) => (
                                    <tr key={v.id || idx}>
                                        <td>{formatTime(v.timestamp)}</td>
                                        {mode === 'verified' && (
                                            <td>
                                                <span className="badge badge--info">P{v.person_id}</span>
                                            </td>
                                        )}
                                        {mode === 'all' && <td>{v.site_location || '-'}</td>}
                                        <td>{mode === 'verified' ? (v.camera_zone || '-') : (v.camera_id || '-')}</td>
                                        <td>{getViolationBadge(v.violation_type)}</td>
                                        <td>
                                            <span className="badge badge--path">
                                                {v.decision_path || '-'}
                                            </span>
                                        </td>
                                        <td>
                                            {mode === 'verified'
                                                ? (v.judge_confidence?.toFixed(3) || '-')
                                                : (((v.detection_confidence || 0) * 100).toFixed(1) + '%')
                                            }
                                        </td>
                                        {mode === 'verified' && (
                                            <td style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                                                {v.judge_processing_time_ms?.toFixed(0) || '-'}ms
                                            </td>
                                        )}
                                    </tr>
                                ))}
                            </tbody>
                        </table>

                        {/* Pagination */}
                        <div className="pagination">
                            <button
                                className="btn btn--secondary"
                                disabled={page <= 1}
                                onClick={() => setPage(p => p - 1)}
                            >
                                ← Previous
                            </button>
                            <span className="pagination__info">
                                Page {page} of {totalPages}
                            </span>
                            <button
                                className="btn btn--secondary"
                                disabled={page >= totalPages}
                                onClick={() => setPage(p => p + 1)}
                            >
                                Next →
                            </button>
                        </div>
                    </>
                )}
            </div>
        </div>
    )
}

export default HistoryTable
