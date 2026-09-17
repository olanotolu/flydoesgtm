# Clay draft integration

Workspace: `Concya` (`1367588`)

The authenticated workspace contains the draft workflow:

- Name: `Fly Brain GTM — NO SEND`
- Workflow: `wf_0tlhgwhtBWfryDomFGR`
- Routine: `workflow:wf_0tlhgwhtBWfryDomFGR`
- Manual trigger: `9d59edee-88f4-43dc-a6c3-91d2a7403dd7`
- URL: https://app.clay.com/workspaces/1367588/terracotta/tc-workflows/wf_0tlhgwhtBWfryDomFGR

The trigger accepts `company` and `domain`. The graph contains four explicitly
named research placeholders for Enrich Company, Company Job Openings, Company
News, and Website Location Research. It is intentionally not published while
the workspace's workflow beta configuration does not expose the managed-action
node schemas through the current CLI. No run is started from this repository.

The local adapter is the enforcement boundary: it permits only read/enrichment
routines, rejects send/reply/enroll/CRM/campaign verbs, caps records and
credits, and never creates recipients or a sender. Once the graph is configured
in Clay, set `CLAYFLY_ROUTINE_ID=workflow:wf_0tlhgwhtBWfryDomFGR` and keep
`CLAYFLY_LIVE_ROUTINES=1` for a live-draft smoke run.

Campaigns/Audiences access was unavailable in this workspace during setup, so
no campaign was created. The local artifact below is the inactive template for
the requested presentation object; it is not connected to Clay and cannot send.
