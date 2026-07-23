export function EmptyState({
  title,
  body,
}: {
  title: string;
  body: string;
}) {
  return (
    <div className="empty-state">
      <span className="empty-index">00</span>
      <h3>{title}</h3>
      <p>{body}</p>
    </div>
  );
}
