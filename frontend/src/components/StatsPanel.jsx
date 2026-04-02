/**
 * StatsPanel Component
 * Displays detection and pipeline statistics
 */

const StatsPanel = ({ stats, timing, pipelineStats }) => {
    // Pipeline mode — show Sentry + Judge stats
    if (pipelineStats) {
        const { sentry, judge, bypass_rate } = pipelineStats

        return (
            <div className="card">
                <div className="card__header">
                    <h3 className="card__title">📊 Pipeline Statistics</h3>
                </div>

                {/* Sentry Section */}
                <div style={{ marginBottom: '1rem' }}>
                    <h4 style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        🛡️ Sentry (YOLO + ByteTrack)
                    </h4>
                    <div className="stats-panel">
                        <div className="stat-item">
                            <div className="stat-item__value">{sentry.effective_fps}</div>
                            <div className="stat-item__label">FPS</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-item__value">{sentry.unique_persons}</div>
                            <div className="stat-item__label">Persons Tracked</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-item__value">{sentry.frames_processed}</div>
                            <div className="stat-item__label">Frames</div>
                        </div>
                        <div className={`stat-item stat-item--${bypass_rate >= 70 ? 'safe' : 'warning'}`}>
                            <div className="stat-item__value">{bypass_rate}%</div>
                            <div className="stat-item__label">SAM Bypass</div>
                        </div>
                    </div>
                </div>

                {/* Judge Section */}
                <div style={{ marginBottom: '1rem' }}>
                    <h4 style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        ⚖️ Judge (SAM 3 Verification)
                    </h4>
                    <div className="stats-panel">
                        <div className={`stat-item stat-item--${judge.confirmed > 0 ? 'violation' : 'safe'}`}>
                            <div className="stat-item__value">{judge.confirmed}</div>
                            <div className="stat-item__label">Confirmed</div>
                        </div>
                        <div className="stat-item stat-item--safe">
                            <div className="stat-item__value">{judge.rejected}</div>
                            <div className="stat-item__label">Rejected</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-item__value">{judge.not_person_rejected}</div>
                            <div className="stat-item__label">Not Person</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-item__value">{judge.avg_time_ms}ms</div>
                            <div className="stat-item__label">Avg SAM Time</div>
                        </div>
                    </div>
                </div>

                {/* Efficiency Metrics */}
                <div style={{ marginBottom: '0.5rem' }}>
                    <h4 style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        📈 Efficiency
                    </h4>
                    <div className="stats-panel">
                        <div className="stat-item">
                            <div className="stat-item__value">{sentry.violations_queued}</div>
                            <div className="stat-item__label">Queued</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-item__value">{sentry.cooldown_skipped}</div>
                            <div className="stat-item__label">Cooldown Skips</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-item__value">{sentry.filtered_false_persons}</div>
                            <div className="stat-item__label">False Persons</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-item__value">{judge.confirmation_rate}%</div>
                            <div className="stat-item__label">Confirm Rate</div>
                        </div>
                    </div>
                </div>

                {/* Path Distribution */}
                {sentry.path_distribution && Object.keys(sentry.path_distribution).length > 0 && (
                    <div className="mt-md">
                        <h4 style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            🔀 Decision Paths
                        </h4>
                        {Object.entries(sentry.path_distribution).map(([path, count]) => (
                            <div key={path} className="flex justify-between mb-sm" style={{ fontSize: '0.8rem' }}>
                                <span>{path}</span>
                                <span className="badge badge--info">{count}</span>
                            </div>
                        ))}
                    </div>
                )}

                {judge.sam_mock_mode && (
                    <div className="mt-sm" style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                        ⚠️ SAM running in mock mode (no GPU)
                    </div>
                )}
            </div>
        )
    }

    // Original mode — single image detection stats
    if (!stats) {
        return (
            <div className="card">
                <div className="card__header">
                    <h3 className="card__title">📊 Statistics</h3>
                </div>
                <div className="stats-panel">
                    <div className="stat-item">
                        <div className="stat-item__value">-</div>
                        <div className="stat-item__label">Persons</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-item__value">-</div>
                        <div className="stat-item__label">Violations</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-item__value">-</div>
                        <div className="stat-item__label">Compliance</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-item__value">-</div>
                        <div className="stat-item__label">SAM Calls</div>
                    </div>
                </div>
            </div>
        )
    }

    const complianceColor = stats.compliance_rate >= 80 ? 'safe' : 'violation'

    return (
        <div className="card">
            <div className="card__header">
                <h3 className="card__title">📊 Statistics</h3>
            </div>

            <div className="stats-panel">
                <div className="stat-item">
                    <div className="stat-item__value">{stats.total_persons}</div>
                    <div className="stat-item__label">Persons</div>
                </div>

                <div className={`stat-item stat-item--${stats.total_violations > 0 ? 'violation' : 'safe'}`}>
                    <div className="stat-item__value">{stats.total_violations}</div>
                    <div className="stat-item__label">Violations</div>
                </div>

                <div className={`stat-item stat-item--${complianceColor}`}>
                    <div className="stat-item__value">{stats.compliance_rate.toFixed(0)}%</div>
                    <div className="stat-item__label">Compliance</div>
                </div>

                <div className="stat-item">
                    <div className="stat-item__value">{stats.sam_activations}</div>
                    <div className="stat-item__label">SAM Calls</div>
                </div>
            </div>

            {/* Timing breakdown */}
            {timing && (
                <div className="mt-md">
                    <h4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                        Processing Time
                    </h4>
                    <div className="progress-bar">
                        <div
                            className="progress-bar__fill"
                            style={{ width: `${Math.min((timing.yolo_ms / timing.total_ms) * 100, 100)}%` }}
                        ></div>
                    </div>
                    <div className="flex justify-between mt-sm" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        <span>YOLO: {timing.yolo_ms.toFixed(1)}ms</span>
                        <span>SAM: {timing.sam_ms.toFixed(1)}ms</span>
                        <span>Total: {timing.total_ms.toFixed(1)}ms</span>
                    </div>
                </div>
            )}

            {/* Path distribution */}
            {stats.path_distribution && (
                <div className="mt-md">
                    <h4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>
                        Decision Paths
                    </h4>
                    {Object.entries(stats.path_distribution).map(([path, count]) => (
                        <div key={path} className="flex justify-between mb-sm" style={{ fontSize: '0.8rem' }}>
                            <span>{path}</span>
                            <span className="badge badge--info">{count}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}

export default StatsPanel
