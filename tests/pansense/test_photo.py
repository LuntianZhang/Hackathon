from pansense.photo import reliable_proposals

def test_low_margin_background_and_edge_glare_are_rejected_without_dropping_confident_partial_food():
    proposals=[{'bbox':[0,0,99,59]},{'bbox':[30,50,20,9]},
               {'bbox':[20,20,15,15]},{'bbox':[0,15,30,30]}]
    scores=[{'similarity':.27,'margin':.008},{'similarity':.27,'margin':.014},
            {'similarity':.28,'margin':.011},{'similarity':.35,'margin':.09}]
    kept=reliable_proposals(proposals,scores,100,60)
    assert [entry[0] for entry in kept]==[proposals[2],proposals[3]]
