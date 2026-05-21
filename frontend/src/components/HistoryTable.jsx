/**
 * Violation History — dashboard layout
 * Summary cards · filter bar · evidence table with expandable ROI rows
 */

import { useState, useEffect, useCallback } from 'react'
import toast from 'react-hot-toast'
import { getHistory, getHistorySummary, getVerifiedViolations, generateReport, listReports } from '../api/client'

function toImgUrl(path) {
    if (!path) return null
    if (path.startsWith('/uploads/') || path.startsWith('/reports/')) return path
    const norm = path.replace(/\\/g, '/')
    const idx = norm.lastIndexOf('/uploads/')
    if (idx !== -1) return norm.slice(idx)
    return `/uploads/${norm.split('/').pop()}`
}

const VIOLATION_META = {
    no_helmet: { label: 'No Helmet', color: '#f59e0b', bg: '#f59e0b20' },
    no_vest: { label: 'No Vest', color: '#f59e0b', bg: '#f59e0b20' },
    both_missing: { label: 'Both Missing', color: '#ef4444', bg: '#ef444420' },
}

function ViolationBadge({ type }) {
    const m = VIOLATION_META[type] || { label: type, color: '#6b7280', bg: '#6b728020' }
    return (
        <span style={{
            background: m.bg, color: m.color,
            padding: '2px 10px', borderRadius: 20,
            fontSize: '0.75rem', fontWeight: 700, whiteSpace: 'nowrap'
        }}>
            {m.label}
        </span>
    )
}

function StatCard({ value, label, color, sub }) {
    return (
        <div className="card" style={{ textAlign: 'center', padding: '1.25rem 1rem' }}>
            <div style={{ fontSize: '2rem', fontWeight: 800, color: color || 'var(--text-primary)', lineHeight: 1 }}>
                {value ?? '-'}
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.4rem', fontWeight: 500 }}>
                {label}
            </div>
            {sub && <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>{sub}</div>}
        </div>
    )
}

function HistoryTable() {
    const [violations, setViolations] = useState([])
    const [summary, setSummary] = useState(null)
    const [isLoading, setIsLoading] = useState(true)
    const [error, setError] = useState(null)
    const [mode, setMode] = useState('all')
    const [dateFilter, setDateFilter] = useState('')
    const [typeFilter, setTypeFilter] = useState('')
    const [page, setPage] = useState(1)
    const [totalPages, setTotalPages] = useState(1)
    const [expandedId, setExpandedId] = useState(null)
    const [refreshTs, setRefreshTs] = useState(Date.now())
    const pageSize = 10

    const fetchViolations = useCallback(async () => {
        setIsLoading(true)
        setError(null)
        try {
            if (mode === 'verified') {
                const params = { limit: pageSize, offset: (page - 1) * pageSize }
                if (typeFilter) params.violation_type = typeFilter
                const data = await getVerifiedViolations(params)
                setViolations(data.violations || [])
                setTotalPages(Math.ceil((data.total_count || 0) / pageSize))
            } else {
                const params = { page, page_size: pageSize }
                if (dateFilter) { params.date_from = dateFilter; params.date_to = dateFilter }
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
    }, [page, dateFilter, typeFilter, mode, refreshTs])

    const fetchSummary = useCallback(async () => {
        try {
            const data = await getHistorySummary(7)
            setSummary(data)
        } catch (err) {
            console.error('Failed to fetch summary:', err)
        }
    }, [refreshTs])

    useEffect(() => { fetchViolations() }, [fetchViolations])
    useEffect(() => { fetchSummary() }, [fetchSummary])
    useEffect(() => { setPage(1) }, [dateFilter, typeFilter, mode])

    const refresh = () => { setRefreshTs(Date.now()) }

    const complianceColor = summary?.compliance_rate >= 80 ? '#10b981' : '#ef4444'

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

            {/* ── Summary cards ───────────────────────────────────────── */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '1rem' }}>
                <StatCard
                    value={summary?.total_violations ?? '-'}
                    label="Total Violations"
                    sub="Last 7 days"
                    color="#ef4444"
                />
                <StatCard
                    value={summary?.compliance_rate != null ? `${summary.compliance_rate.toFixed(1)}%` : '-'}
                    label="Compliance Rate"
                    color={complianceColor}
                />
                <StatCard
                    value={summary?.no_helmet_count ?? '-'}
                    label="No Helmet"
                    color="#f59e0b"
                />
                <StatCard
                    value={summary?.no_vest_count ?? '-'}
                    label="No Vest"
                    color="#f59e0b"
                />
                <StatCard
                    value={summary?.both_missing_count ?? '-'}
                    label="Both Missing"
                    color="#ef4444"
                />
                <StatCard
                    value={summary?.sam_activations ?? '-'}
                    label="SAM Activations"
                    color="var(--color-primary)"
                />
            </div>

            {/* ── Filter bar ──────────────────────────────────────────── */}
            <div className="card" style={{ padding: '1rem' }}>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                    {/* Mode toggle */}
                    <div style={{ display: 'flex', background: 'var(--bg-secondary)', borderRadius: 8, padding: 2, gap: 2 }}>
                        {['all', 'verified'].map(m => (
                            <button
                                key={m}
                                className={`btn btn--sm ${mode === m ? 'btn--primary' : 'btn--ghost'}`}
                                style={{ borderRadius: 6, fontSize: '0.8rem', padding: '0.25rem 0.9rem' }}
                                onClick={() => setMode(m)}
                            >
                                {m === 'all' ? '📋 All' : '⚖️ Verified'}
                            </button>
                        ))}
                    </div>

                    {mode === 'all' && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Date</label>
                            <input
                                type="date"
                                value={dateFilter}
                                onChange={e => setDateFilter(e.target.value)}
                                className="input"
                                style={{ fontSize: '0.85rem', padding: '0.3rem 0.6rem' }}
                            />
                        </div>
                    )}

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <label style={{ fontSize: '0.8rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>Type</label>
                        <select
                            value={typeFilter}
                            onChange={e => setTypeFilter(e.target.value)}
                            className="input"
                            style={{ fontSize: '0.85rem', padding: '0.3rem 0.6rem' }}
                        >
                            <option value="">All Types</option>
                            <option value="no_helmet">No Helmet</option>
                            <option value="no_vest">No Vest</option>
                            <option value="both_missing">Both Missing</option>
                        </select>
                    </div>

                    <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem' }}>
                        {(dateFilter || typeFilter) && (
                            <button className="btn btn--secondary btn--sm" onClick={() => { setDateFilter(''); setTypeFilter('') }}>
                                Clear
                            </button>
                        )}
                        <button className="btn btn--secondary btn--sm" onClick={refresh} title="Refresh">
                            🔄 Refresh
                        </button>
                    </div>
                </div>
            </div>

            {/* ── Table ───────────────────────────────────────────────── */}
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                <div style={{ padding: '1rem 1.25rem', borderBottom: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 style={{ margin: 0, fontSize: '1rem' }}>
                        {mode === 'verified' ? '⚖️ Verified Violations (Judge-Confirmed)' : '📋 Violation History'}
                    </h3>
                    {!isLoading && (
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                            {violations.length} record{violations.length !== 1 ? 's' : ''}
                        </span>
                    )}
                </div>

                {isLoading ? (
                    <div style={{ padding: '3rem', textAlign: 'center' }}>
                        <span className="loading-spinner" style={{ width: 32, height: 32 }} />
                        <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Loading violations...</p>
                    </div>
                ) : error ? (
                    <div style={{ padding: '2rem', textAlign: 'center' }}>
                        <p style={{ color: '#ef4444', marginBottom: '1rem' }}>❌ {error}</p>
                        <button className="btn btn--primary" onClick={refresh}>Retry</button>
                    </div>
                ) : violations.length === 0 ? (
                    <div style={{ padding: '3rem', textAlign: 'center' }}>
                        <div style={{ fontSize: '3rem', opacity: 0.2, marginBottom: '0.5rem' }}>📭</div>
                        <p style={{ color: 'var(--text-muted)' }}>
                            {mode === 'verified' ? 'No verified violations yet. Run the pipeline first.' : 'No violations found.'}
                        </p>
                    </div>
                ) : (
                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                            <thead>
                                <tr style={{ background: 'var(--bg-secondary)', borderBottom: '2px solid var(--border-color)' }}>
                                    <th style={{ padding: '0.75rem 1rem', textAlign: 'left', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em', width: 70 }}>Evidence</th>
                                    <th style={{ padding: '0.75rem 1rem', textAlign: 'left', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Time</th>
                                    {mode === 'verified' && <th style={{ padding: '0.75rem 1rem', textAlign: 'left', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Person</th>}
                                    {mode === 'all' && <th style={{ padding: '0.75rem 1rem', textAlign: 'left', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Site</th>}
                                    <th style={{ padding: '0.75rem 1rem', textAlign: 'left', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Camera</th>
                                    <th style={{ padding: '0.75rem 1rem', textAlign: 'left', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Violation</th>
                                    <th style={{ padding: '0.75rem 1rem', textAlign: 'left', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Path</th>
                                    <th style={{ padding: '0.75rem 1rem', textAlign: 'right', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Confidence</th>
                                    {mode === 'all' && <th style={{ padding: '0.75rem 1rem', textAlign: 'center', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>SAM</th>}
                                </tr>
                            </thead>
                            <tbody>
                                {violations.map((v, idx) => {
                                    const roiUrl = toImgUrl(v.cropped_roi_path || v.image_path)
                                    const annotatedUrl = toImgUrl(v.annotated_image_path)
                                    const key = v.id || idx
                                    const isExpanded = expandedId === key
                                    const hasEvidence = roiUrl || annotatedUrl

                                    return (
                                        <>
                                            <tr
                                                key={key}
                                                onClick={() => hasEvidence && setExpandedId(isExpanded ? null : key)}
                                                style={{
                                                    borderBottom: '1px solid var(--border-color)',
                                                    cursor: hasEvidence ? 'pointer' : 'default',
                                                    background: isExpanded ? 'var(--bg-secondary)' : 'transparent',
                                                    transition: 'background 0.15s',
                                                }}
                                                onMouseEnter={e => { if (!isExpanded) e.currentTarget.style.background = 'var(--bg-secondary)40' }}
                                                onMouseLeave={e => { if (!isExpanded) e.currentTarget.style.background = 'transparent' }}
                                            >
                                                {/* Evidence thumbnail */}
                                                <td style={{ padding: '0.6rem 1rem' }}>
                                                    {roiUrl ? (
                                                        <img
                                                            src={roiUrl}
                                                            alt="ROI"
                                                            style={{ width: 40, height: 54, objectFit: 'cover', borderRadius: 4, border: '1px solid var(--border-color)', display: 'block' }}
                                                            onError={e => { e.target.style.display = 'none' }}
                                                        />
                                                    ) : annotatedUrl ? (
                                                        <span style={{ fontSize: '1.3rem' }}>🖼️</span>
                                                    ) : (
                                                        <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>—</span>
                                                    )}
                                                </td>

                                                <td style={{ padding: '0.6rem 1rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                                                    {v.timestamp ? new Date(v.timestamp).toLocaleString() : '-'}
                                                </td>

                                                {mode === 'verified' && (
                                                    <td style={{ padding: '0.6rem 1rem' }}>
                                                        <span style={{ background: 'var(--color-primary)20', color: 'var(--color-primary)', padding: '2px 8px', borderRadius: 12, fontSize: '0.78rem', fontWeight: 600 }}>
                                                            P{v.person_id}
                                                        </span>
                                                    </td>
                                                )}
                                                {mode === 'all' && (
                                                    <td style={{ padding: '0.6rem 1rem', color: 'var(--text-secondary)' }}>{v.site_location || '-'}</td>
                                                )}

                                                <td style={{ padding: '0.6rem 1rem', color: 'var(--text-secondary)' }}>
                                                    {mode === 'verified' ? (v.camera_zone || '-') : (v.camera_id || '-')}
                                                </td>

                                                <td style={{ padding: '0.6rem 1rem' }}>
                                                    <ViolationBadge type={v.violation_type} />
                                                </td>

                                                <td style={{ padding: '0.6rem 1rem' }}>
                                                    <span style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', padding: '2px 8px', borderRadius: 8, fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                                                        {v.decision_path || '-'}
                                                    </span>
                                                </td>

                                                <td style={{ padding: '0.6rem 1rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                                                    {mode === 'verified'
                                                        ? (v.judge_confidence?.toFixed(3) ?? '-')
                                                        : `${((v.detection_confidence || 0) * 100).toFixed(1)}%`
                                                    }
                                                </td>

                                                {mode === 'all' && (
                                                    <td style={{ padding: '0.6rem 1rem', textAlign: 'center' }}>
                                                        <span style={{ fontSize: '0.75rem', color: v.sam_activated ? '#f59e0b' : 'var(--text-muted)' }}>
                                                            {v.sam_activated ? '✅' : '—'}
                                                        </span>
                                                    </td>
                                                )}
                                            </tr>

                                            {/* Expanded evidence row */}
                                            {isExpanded && (
                                                <tr key={`${key}-exp`} style={{ borderBottom: '1px solid var(--border-color)' }}>
                                                    <td colSpan={mode === 'verified' ? 8 : 8} style={{ padding: '1.25rem 1rem', background: 'var(--bg-secondary)' }}>
                                                        <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                                                            {roiUrl && (
                                                                <div>
                                                                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Person ROI</div>
                                                                    <img src={roiUrl} alt="ROI" style={{ maxHeight: 220, borderRadius: 8, border: '1px solid var(--border-color)', display: 'block' }} />
                                                                </div>
                                                            )}
                                                            {annotatedUrl && (
                                                                <div>
                                                                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Annotated Frame</div>
                                                                    <img src={annotatedUrl} alt="Annotated" style={{ maxHeight: 220, borderRadius: 8, border: '1px solid var(--border-color)', display: 'block' }} />
                                                                </div>
                                                            )}
                                                            <div style={{ fontSize: '0.85rem', lineHeight: 2 }}>
                                                                <div><strong>Violation:</strong> {VIOLATION_META[v.violation_type]?.label || v.violation_type}</div>
                                                                <div><strong>Decision Path:</strong> {v.decision_path || '-'}</div>
                                                                <div><strong>Confidence:</strong> {mode === 'verified' ? v.judge_confidence?.toFixed(3) : `${((v.detection_confidence || 0) * 100).toFixed(1)}%`}</div>
                                                                {mode === 'all' && <div><strong>SAM Activated:</strong> {v.sam_activated ? 'Yes' : 'No'}</div>}
                                                                {v.occurrence_count > 1 && <div><strong>Re-detections:</strong> {v.occurrence_count}</div>}
                                                                {v.total_duration_minutes > 0 && <div><strong>Duration:</strong> {v.total_duration_minutes.toFixed(1)} min</div>}
                                                            </div>
                                                        </div>
                                                    </td>
                                                </tr>
                                            )}
                                        </>
                                    )
                                })}
                            </tbody>
                        </table>
                    </div>
                )}

                {/* Pagination */}
                {totalPages > 1 && !isLoading && (
                    <div style={{ padding: '0.75rem 1.25rem', borderTop: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <button className="btn btn--secondary btn--sm" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>
                            ← Previous
                        </button>
                        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                            Page {page} of {totalPages}
                        </span>
                        <button className="btn btn--secondary btn--sm" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>
                            Next →
                        </button>
                    </div>
                )}
            </div>

            {/* ── Report Generator ──────────────────────────────────── */}
            <ReportPanel />
        </div>
    )
}

// ── Report Generator Panel ───────────────────────────────────────────────────
function ReportPanel() {
    const today = new Date().toISOString().slice(0, 10)
    const [dateFrom, setDateFrom]   = useState(today)
    const [dateTo,   setDateTo]     = useState(today)
    const [sending,  setSending]    = useState(false)
    const [sendEmail, setSendEmail] = useState(false)
    const [email,    setEmail]      = useState('')
    const [lastReport, setLastReport] = useState(null)
    const [pastReports, setPastReports] = useState([])
    const [showPast, setShowPast]   = useState(false)

    const fetchPast = useCallback(async () => {
        try {
            const data = await listReports()
            setPastReports(data.reports || [])
        } catch { /* silent */ }
    }, [])

    useEffect(() => { fetchPast() }, [fetchPast])

    const handleGenerate = async () => {
        if (!dateFrom || !dateTo) { toast.error('Select both dates'); return }
        if (sendEmail && !email.trim()) { toast.error('Enter recipient email'); return }
        setSending(true)
        const toastId = toast.loading('Generating report…')
        try {
            const data = await generateReport(dateFrom, dateTo, {
                sendEmail,
                email: email.trim() || undefined,
            })
            setLastReport(data)
            fetchPast()
            const emailMsg = sendEmail
                ? ` · email: ${data.email_status}`
                : ''
            toast.success(`Report ready — ${data.stats.total_violations} violations${emailMsg}`, { id: toastId })
        } catch (err) {
            toast.error(err.response?.data?.detail || 'Report generation failed', { id: toastId })
        } finally {
            setSending(false)
        }
    }

    return (
        <div className="card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ margin: 0, fontSize: '1rem' }}>📄 Generate Report</h3>
                <button
                    className="btn btn--ghost btn--sm"
                    onClick={() => { setShowPast(p => !p); fetchPast() }}
                    style={{ fontSize: '0.8rem' }}
                >
                    {showPast ? 'Hide' : 'Past Reports'}
                </button>
            </div>

            {/* Date range + options */}
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>From</label>
                    <input
                        type="date" value={dateFrom} max={today}
                        onChange={e => setDateFrom(e.target.value)}
                        className="input" style={{ fontSize: '0.85rem', padding: '0.35rem 0.6rem' }}
                    />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>To</label>
                    <input
                        type="date" value={dateTo} min={dateFrom} max={today}
                        onChange={e => setDateTo(e.target.value)}
                        className="input" style={{ fontSize: '0.85rem', padding: '0.35rem 0.6rem' }}
                    />
                </div>

                {/* Email toggle */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                    <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Send via email</label>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', height: 36 }}>
                        <input
                            type="checkbox" id="send-email-chk"
                            checked={sendEmail} onChange={e => setSendEmail(e.target.checked)}
                            style={{ width: 16, height: 16, cursor: 'pointer' }}
                        />
                        <label htmlFor="send-email-chk" style={{ fontSize: '0.85rem', cursor: 'pointer' }}>Yes</label>
                    </div>
                </div>

                {sendEmail && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', flex: 1, minWidth: 200 }}>
                        <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Recipient email</label>
                        <input
                            type="email" value={email}
                            onChange={e => setEmail(e.target.value)}
                            placeholder="manager@site.com"
                            className="input" style={{ fontSize: '0.85rem', padding: '0.35rem 0.6rem' }}
                        />
                    </div>
                )}

                <button
                    className="btn btn--primary"
                    onClick={handleGenerate}
                    disabled={sending}
                    style={{ alignSelf: 'flex-end' }}
                >
                    {sending
                        ? <><span className="loading-spinner" style={{ width: 14, height: 14 }} /> Generating…</>
                        : '📄 Generate PDF'}
                </button>
            </div>

            {/* Latest report download */}
            {lastReport?.pdf_url && (
                <div style={{
                    marginTop: '1rem', padding: '0.75rem 1rem',
                    background: '#10b98115', border: '1px solid #10b98140',
                    borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem'
                }}>
                    <div>
                        <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>✅ Report Ready</div>
                        <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 2 }}>
                            {lastReport.stats.total_violations} violations · {lastReport.stats.compliance_rate?.toFixed(1) ?? '—'}% compliance
                            {lastReport.email_status && lastReport.email_status !== 'not_requested' && (
                                <span style={{ marginLeft: 8 }}>· 📧 {lastReport.email_status}</span>
                            )}
                        </div>
                    </div>
                    <a
                        href={lastReport.pdf_url}
                        download
                        className="btn btn--secondary"
                        style={{ fontSize: '0.85rem' }}
                    >
                        📥 Download PDF
                    </a>
                </div>
            )}

            {/* Past reports list */}
            {showPast && pastReports.length > 0 && (
                <div style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }}>
                        Previously Generated
                    </div>
                    {pastReports.map(r => (
                        <div key={r.id} style={{
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                            padding: '0.5rem 0.75rem', background: 'var(--bg-secondary)',
                            borderRadius: 6, fontSize: '0.85rem', flexWrap: 'wrap', gap: '0.5rem'
                        }}>
                            <div>
                                <span style={{ fontWeight: 600 }}>{r.report_date}</span>
                                <span style={{ color: 'var(--text-muted)', marginLeft: 12 }}>
                                    {r.total_violations} violations · {r.compliance_rate?.toFixed(1)}% compliance
                                </span>
                                {r.email_sent && <span style={{ marginLeft: 8, color: '#10b981', fontSize: '0.78rem' }}>📧 sent</span>}
                            </div>
                            {r.pdf_url && (
                                <a href={r.pdf_url} download className="btn btn--ghost btn--sm" style={{ fontSize: '0.78rem' }}>
                                    📥 PDF
                                </a>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {showPast && pastReports.length === 0 && (
                <p style={{ marginTop: '0.75rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    No reports generated yet.
                </p>
            )}
        </div>
    )
}

export default HistoryTable
