# Wave-7 Lane A — ship notes

Things a release note and a rollout have to say out loud. Everything else is in the
commit trail; this file carries only what changes for someone OPERATING the connector.

## Behavioural change: `build_communities` no longer clears other partitions

**What changed.** `Graphiti.build_communities(group_ids=[...])` used to delete every
Community node in the store before rebuilding the partitions it was asked for. It now
deletes only the requested partitions' communities (BUG-57).

**This narrows the blast radius, and narrowing is a behaviour change.** For any
deployment where several `group_id`s share one database — every AGE deployment, since
AGE keeps all partitions in one graph — communities belonging to the group_ids you did
NOT name now SURVIVE a rebuild. That is the point of the fix. But it means the store's
state after a rebuild differs from before:

| | before | after |
|---|---|---|
| communities in the named group_ids | deleted, rebuilt | deleted, rebuilt |
| communities in every other group_id | **deleted, never rebuilt** | **left intact** |

So anything that relied — knowingly or not — on `build_communities` as a way to clear
the whole community layer no longer gets that. Use `group_ids=None` for a genuine
whole-graph rebuild; it still clears everything.

On FalkorDB the practical difference is smaller, because Graphiti maps `group_id` to a
separate FalkorDB database and the per-group fan-out already confined most of the
damage to one graph. On AGE, and on any single-database backend, the old behaviour was
destroying live data on every scoped rebuild.

**Stale communities are now possible where they were not.** Previously a rebuild left
every other partition with NO communities (destroyed). Now it leaves them with their
PREVIOUS ones. A partition whose entities have changed since its last rebuild will
carry communities that are out of date rather than absent. That is the correct
trade — absent-and-silently-destroyed is worse than stale-and-rebuildable — but a
rollout should expect it and rebuild per partition where freshness matters.

## `group_ids=[]` is now a no-op

`remove_communities(driver, [])` and `build_communities(group_ids=[])` delete NOTHING.
They previously cleared the whole graph. Anything scripted that passed an empty list
expecting a full clear must pass `None` instead.

This was not a cosmetic argument change: `get_community_clusters` iterates the list, so
an empty one builds nothing. `build_communities([])` therefore used to delete every
community and rebuild none of them.

## Connector version

`mcp_server/pyproject.toml` is bumped to **1.6.1** (all-fix patch). It is served in
three places that all read the packaged value — `get_status.version`, the instructions
header, and `serverInfo.version` — so every recreated connector should announce 1.6.1
after rollout. That is the cheapest post-rollout check that the new image is live.

## Contract text moved with the code

The wire description and the README entry for `build_communities` now describe the
scoped delete. A consumer that cached the old announcement will be carrying a claim
that overstates the blast radius until it re-reads the instructions.
