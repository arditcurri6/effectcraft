use super::*;

fn st() -> EditorState {
    let look = Look::new("L", vec![Node::new(ToolKind::Exposure), Node::new(ToolKind::Contrast)]);
    EditorState { baseline: Some(Snapshot { look: look.clone(), strength: 100.0, selected: None }), draft: look, strength: 100.0, ..Default::default() }
}

#[test]
fn edits_undo_and_redo() {
    let mut s = st();
    assert!(!s.modified());
    s.set_param(0, "exposure", 1.0);
    s.set_param(1, "contrast", 20.0);
    assert!(s.modified());
    assert_eq!(s.draft.nodes[1].num("contrast"), 20.0);
    s.undo();
    assert_eq!(s.draft.nodes[1].num("contrast"), 0.0);
    assert_eq!(s.draft.nodes[0].num("exposure"), 1.0);
    s.redo();
    assert_eq!(s.draft.nodes[1].num("contrast"), 20.0);
    // A new edit clears redo.
    s.undo();
    s.set_param(0, "exposure", 2.0);
    s.redo();
    assert_eq!(s.draft.nodes[1].num("contrast"), 0.0);
    // No-op edits add no step.
    let n = s.undo.len();
    s.set_param(0, "exposure", 2.0);
    assert_eq!(s.undo.len(), n);
}

#[test]
fn a_gesture_is_one_undo_step() {
    let mut s = st();
    for v in [0.1, 0.2, 0.3, 0.4] {
        s.gesture_edit(|l, _| {
            l.nodes[0].params.insert("exposure".into(), v.into());
        });
    }
    assert_eq!(s.undo.len(), 1);
    s.gesture = false;
    s.undo();
    assert_eq!(s.draft.nodes[0].num("exposure"), 0.0);
}

#[test]
fn undo_keeps_the_selection_in_range() {
    let mut s = st();
    s.selected = Some(1);
    s.edit(|l, _| {
        l.nodes.push(Node::new(ToolKind::FilmGrain));
    });
    s.selected = Some(2);
    s.undo();
    assert!(s.selected.is_none_or(|i| i < s.draft.nodes.len()));
}
