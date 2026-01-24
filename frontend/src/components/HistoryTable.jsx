/**
 * History Table Component
 * 
 * Displays violation history with filtering and pagination.
 */

import { useState, useEffect, useCallback } from 'react'
import { getHistory, getHistorySummary } from '../api/client'

function HistoryTable() {
    // State
    const [violations, setViolations] = useState([])
    const [summary, setSummary] = useState(null)
    const [isLoading, setIsLoading] = useState(true)
    const [error, setError] = useState(null)

    // Filters
    const [dateFilter, setDateFilter] = useState('')
    const [typeFilter, setTypeFilter] = useState('')

    // Pagination
    const [page, setPage] = useState(1)
    const [totalPages, setTotalPages] = useState(1)
    const pageSize = 10

    // Fetch violations
    const fetchViolations = useCallback(async () => {
        setIsLoading(true)
        setError(null)

        try {
            const params = {
                page,
                page_size: pageSize
            }

            if (dateFilter) {
                params.date_from = dateFilter
                params.date_to = dateFilter
            }

            if (typeFilter) {
                params.violation_type = typeFilter
            }

            const data = await getHistory(params)
            setViolations(data.violations || [])
            setTotalPages(Math.ceil((data.total || 0) / pageSize))

        } catch (err) {
            console.error('Failed to fetch history:', err)
            setError('Failed to load violation history')
        } finally {
            setIsLoading(false)
        }
    }, [page, dateFilter, typeFilter])

    // Fetch summary
    const fetchSummary = useCallback(async () => {
        try {
            const data = await getHistorySummary(7)
            setSummary(data)
        } catch (err) {
            console.error('Failed to fetch summary:', err)
        }
    }, [])

    // Load data on mount and filter changes
    useEffect(() => {
        fetchViolations()
    }, [fetchViolations])

    useEffect(() => {
        fetchSummary()
    }, [fetchSummary])

    // Reset to page 1 when filters change
    useEffect(() => {
        setPage(1)
    }, [dateFilter, typeFilter])

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

            {/* Filters */}
            <div className="card filters-card">
                <h4>🔍 Filters</h4>
                <div className="filters-row">
                    <div className="filter-group">
                        <label>Date</label>
                        <input
                            type="date"
                            value={dateFilter}
                            onChange={(e) => setDateFilter(e.target.value)}
                            className="input"
                        />
                    </div>
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
                <h4>📋 Violation History</h4>

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
                        <p>No violations found</p>
                    </div>
                ) : (
                    <>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Site</th>
                                    <th>Camera</th>
                                    <th>Type</th>
                                    <th>Path</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {violations.map((v, idx) => (
                                    <tr key={v.id || idx}>
                                        <td>{formatTime(v.timestamp)}</td>
                                        <td>{v.site_location || '-'}</td>
                                        <td>{v.camera_id || '-'}</td>
                                        <td>{getViolationBadge(v.violation_type)}</td>
                                        <td>
                                            <span className="badge badge--path">
                                                {v.decision_path || '-'}
                                            </span>
                                        </td>
                                        <td>{((v.detection_confidence || 0) * 100).toFixed(1)}%</td>
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
